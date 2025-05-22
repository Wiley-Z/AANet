import logging
import pickle as pkl
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union
import lmdb
import time
import zstandard as zstd
from unicore.data import BaseWrapperDataset
import random
import torch
import numpy as np
from copy import deepcopy


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAP_SIZE = 10 * 1024 * 1024 * 1024 * 1024  # 10T
BENCHMARK_SIZE = 2000


class LMDBDataset:
    """
    split:
        full: "key1,key2,..."
        dataset1: "key1,key2,..."
        dataset2: "key2,key3,..."
        train: "key1,key2,..."
        val: "key3,key4,..."
    data:
        key1: pkl.dump({k1: v11, k2: v12, ...})
        key2: pkl.dump({k1: v21, k2: v22, ...})
    """

    SPLIT_DB = "split"
    DATA_DB = "data"
    MAX_CACHE_SIZE = 16
    MAX_WRITE_RETRY_WHEN_LOCK = 10
    RETRY_INTERVAL = 5

    def __init__(
        self,
        lmdb_path: Union[str, Path],
        compressed: bool = True,
        readonly: bool = True,
        enable_cache: bool = True,
    ):
        self.lmdb_path = Path(lmdb_path).resolve()
        self.compressed = compressed
        if self.compressed:
            self.compressor = zstd.ZstdCompressor(level=3)
            self.decompressor = zstd.ZstdDecompressor()
        self.readonly = readonly
        self.enable_cache = enable_cache

        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                self.env = lmdb.open(
                    str(self.lmdb_path),
                    max_dbs=2,
                    map_size=MAP_SIZE,
                    readonly=readonly,
                    lock=not readonly,
                    create=not readonly,
                )
                break
            except Exception as e:
                raise
                time.sleep(self.RETRY_INTERVAL)
        else:
            raise Exception(
                f"Failed to open lmdb after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
            )
        self.db = {
            db_name: self.env.open_db(db_name.encode())
            for db_name in [self.SPLIT_DB, self.DATA_DB]
        }
        self._splits = dict()
        self.default_split = "full"

    def compress(self, content: bytes) -> bytes:
        content = self.compressor.compress(content)
        return content

    def decompress(self, content: bytes) -> bytes:
        content = self.decompressor.decompress(content)
        return content

    def close(self):
        self.env.close()

    def _get_value(self, db: str, key: str, default: Any = None) -> Any:
        # for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
        #     try:
        with self.env.begin(db=self.db[db], write=False) as txn:
            value = txn.get(key.encode())
        #         break
        #     except Exception as e:
        #         time.sleep(self.RETRY_INTERVAL)
        # else:
        #     raise Exception(
        #         f"Failed to read after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
        #     )

        if value is not None:
            if self.compressed:
                value = self.decompress(value)
        else:
            logger.warning(f"Key {key} not found in {db}")
            value = default
        return value

    def _set_value(self, db: str, key: str, value: bytes):
        if self.compressed:
            value = self.compress(value)

        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                with self.env.begin(db=self.db[db], write=True) as txn:
                    txn.put(key.encode(), value)
                return
            except Exception as e:
                time.sleep(self.RETRY_INTERVAL)
        raise Exception(
            f"Failed to write after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
        )

    def _get_values(
        self, db: str, keys: List[str], ori: bool = False
    ) -> List[Any]:
        """better for data construction with large data size."""
        values = []
        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                with self.env.begin(db=self.db[db], write=False) as txn:
                    for key in keys:
                        value = txn.get(key.encode())
                        if value is not None and self.compressed and not ori:
                            value = self.decompress(value)
                        values.append(value)
                return values
            except Exception:
                time.sleep(self.RETRY_INTERVAL)
        raise Exception(
            f"Failed to read after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
        )

    def _set_values(self, db: str, data: Dict[str, Any], ori: bool = False):
        """better for data construction with large data size."""
        compressed = []
        for key, value in data.items():
            if self.compressed and not ori:
                value = self.compress(value)
            compressed.append((key, value))

        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                with self.env.begin(db=self.db[db], write=True) as txn:
                    for key, value in compressed:
                        txn.put(key.encode(), value)
                return
            except Exception:
                time.sleep(self.RETRY_INTERVAL)
        raise Exception(
            f"Failed to write after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
        )

    def _smart_decode_list(self, value: bytes) -> List[Any]:
        try:
            return value.decode().split(",")
        except Exception:
            return pkl.loads(value)

    def _smart_encode_list(self, value: List[Any]) -> bytes:
        if len(value) == 0:
            return pkl.dumps([])
        elif type(value[0]) == str:
            return ",".join(value).encode()
        else:
            return pkl.dumps(value)

    def get_split(self, key: str) -> List[str]:
        if key not in self._splits or not self.enable_cache:
            split_save = self._get_value(self.SPLIT_DB, key)
            if split_save is None:
                return []
            else:
                self._splits[key] = self._smart_decode_list(split_save)
        return self._splits[key]

    def set_split(
        self,
        split: str,
        keys: List[str],
        append: bool = False,
        deduplicate: bool = True,
        update_full: bool = False,
    ):
        if append:
            keys = self.get_split(split) + keys

        if deduplicate:
            keys = list(sorted(list(set(keys))))

        self._set_value(self.SPLIT_DB, split, self._smart_encode_list(keys))
        self._splits[split] = keys
        if (
            split != "full"
            and update_full
            and len(keys) > 0
            and type(keys[0]) == str
        ):
            self.update_full_split()

    def update_full_split(self, from_data: bool = False):
        keys = []
        if from_data:
            with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
                keys = [
                    key.decode() for key in txn.cursor().iternext(values=False)
                ]
        else:  # from split
            with self.env.begin(db=self.db[self.SPLIT_DB], write=False) as txn:
                splits = [
                    key.decode() for key in txn.cursor().iternext(values=False)
                ]
            for split in splits:
                if split != "full":
                    keys.extend(self.get_split(split))
                    keys = list(sorted(list(set(keys))))
        if not self.readonly:
            self.set_split("full", keys)
        else:
            self._splits["full"] = keys

    def check_keys(self):
        with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
            keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        full_keys = self.get_split("full")

        missing_keys = list(set(full_keys) - set(keys))
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")

        orphan_keys = list(set(keys) - set(full_keys))
        if len(orphan_keys) > 0:
            print(f"Orphan keys: {orphan_keys}")

        if len(missing_keys) + len(orphan_keys) == 0:
            print("All keys are in place")

    def __getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        if self.enable_cache:
            return self.__cache_getitem__(key)
        else:
            return self.__imp_getitem__(key)

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def __cache_getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        return self.__imp_getitem__(key)

    def __imp_getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        if isinstance(key, int) or isinstance(key, np.int64):
            key = self.get_split(self.default_split)[key]
        data = self._get_value(self.DATA_DB, key)
        if data is not None:
            data = pkl.loads(data)
        return data

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self._set_value(self.DATA_DB, key, pkl.dumps(value))

    def write_data(
        self, data: Dict[str, Dict[str, Any]], ori: bool = False
    ) -> None:
        """when writing large amount of data, use `write_data` to call
        _set_values instead of __setitem__"""
        if not ori:
            data = {key: pkl.dumps(value) for key, value in data.items()}
        self._set_values(self.DATA_DB, data)

    def get_data(
        self, keys: List[str], ori: bool = False, as_dict: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """when reading large amount of data, use `get_data` to call
        _get_values instead of __getitem__"""
        data = self._get_values(self.DATA_DB, keys)
        if not ori:
            data = [pkl.loads(value) for value in data]
        if as_dict:
            data = dict(zip(keys, data))
        return data

    def __contains__(self, key: str) -> bool:
        with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
            return txn.get(key.encode()) is not None

    def set_default_split(self, split: str) -> None:
        self.default_split = split
        if self.enable_cache:
            self.__cache_getitem__.cache_clear()

    def __len__(self) -> int:
        keys = self.get_split(self.default_split)
        return len(keys)

    def summary(self) -> Dict[str, int]:
        self.update_full_split()
        return {split: len(keys) for split, keys in self._splits.items()}

    def __repr__(self) -> str:
        return f"LMDBDataset({self.lmdb_path})"

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self):
            result = self[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

class ChoiceDataset(BaseWrapperDataset):
    def __init__(self, dataset, filler={"inputs": {"pocket_atoms": ['[PAD]'], "pocket_coordinates": np.zeros((1, 3)), "resi": ['A1']}, "iou": 0}):
        self.dataset = dataset
        self.filler = filler

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if item:
            if isinstance(item, dict):
                return random.choice(list(item.values()))
            return random.choice(item)
        else:
            return deepcopy(self.filler)

class KeyChoiceDataset(BaseWrapperDataset):
    def __init__(self, dataset, keys):
        self.dataset = dataset
        self.keys = keys

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        for key in self.keys:
            item[key] = random.choice(item[key])
        return item


class ListKeyCutoffDataset(BaseWrapperDataset):
    def __init__(self, dataset, key, cutoff=lambda x: x >= 0.5):
        self.dataset = dataset
        self.key = key
        self.cutoff = cutoff

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = [i for i in item if self.cutoff(i[self.key])]
        return item

class ListKeyCatalogChoiceDataset(BaseWrapperDataset):
    def __init__(self, dataset, key='iou', catalog=[(lambda x: x >= 0.5), (lambda x: x <= 0.1)], proportions=None):
        self.dataset = dataset
        self.key = key
        self.catalog = catalog
        self.proportions = proportions
        if self.proportions is None:
            self.proportions = [1 / len(catalog) for _ in range(len(catalog))]
        else:
            assert len(self.proportions) == len(self.catalog), "Proportions must have the same length as the catalog"

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        catalogs = [[i for i in item if catalog(i[self.key])] for catalog in self.catalog]
        proportions = [p if len(catalog) > 0 else 0 for p, catalog in zip(self.proportions, catalogs)]
        if len(catalogs) == 0:
            return None
        else:
            catalog = random.choices(catalogs, weights=self.proportions)[0]
            return catalog

class MappingDataset(BaseWrapperDataset):
    def __init__(self, dataset, mapping=lambda x: x['iou'] >= 0.5, new_key='mask'):
        self.dataset = dataset
        self.mapping = mapping
        self.new_key = new_key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            if self.new_key is None:
                return self.mapping(item)
            item[self.new_key] = self.mapping(item)
            return item
        except Exception as e:
            print(e)
            print(f"type(item): {type(item)}, item: {item}")
            print(f"type(self.new_key): {type(self.new_key)}, self.new_key: {self.new_key}")
            print(f"type(self.mapping): {type(self.mapping)}, self.mapping: {self.mapping}")
            return None


class LMDBKeyDataset(LMDBDataset):
    def __init__(self, lmdb_path, compresed = True, readonly = True, subdir = True):
        super().__init__(lmdb_path, compresed, readonly, subdir)
    
    def __getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        if isinstance(key, int) or isinstance(key, np.int64):
            key = self.get_split(self.default_split)[key]
        return key


class MapItemDataset(BaseWrapperDataset):
    def __init__(self, dataset, map_dataset, idx_func=lambda x: x.rsplit('/', 1)[0], non_exist_func=None):
        self.dataset = dataset
        self.map_dataset = map_dataset
        self.idx_func = idx_func
        self.non_exist_func = non_exist_func

    def __len__(self):
        return len(self.map_dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        idx = self.map_dataset[idx]
        if self.idx_func is not None:
            idx = self.idx_func(idx)
        if idx not in self.dataset and self.non_exist_func is not None:
            return self.non_exist_func(idx)
        item = self.dataset[idx]
        return item

    def collater(self, samples):
        return samples

class StackDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(self.dataset[idx], np.ndarray):
            return np.stack(self.dataset[idx])
        elif isinstance(self.dataset[idx], torch.Tensor):
            return torch.stack(self.dataset[idx])
        else:
            return self.dataset[idx]