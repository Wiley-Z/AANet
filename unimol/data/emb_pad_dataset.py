import numpy as np
import torch
from unicore.data import BaseWrapperDataset

def collate_tokens_values(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if isinstance(values[0], np.ndarray):
        values = [torch.from_numpy(v) for v in values]
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    
    res = values[0].new(len(values), size, values[0].shape[-1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"dts {dst.shape}, {dst.numel()} != {src.shape}, {src.numel()}"
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, :] if left_pad else res[i][: len(v), :])
    return res

class EmbPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False, pad_to_length=None, pad_to_multiple=1):
        """
        Args:
            dataset (BaseWrapperDataset): The dataset to wrap.
            pad_idx (int): The index used for padding.
            left_pad (bool): Whether to pad on the left (default is False, i.e., pad on the right).
            pad_to_length (int, optional): Pad to this length. Default is None, which pads to the longest sequence.
            pad_to_multiple (int, optional): Pad the length to be a multiple of this number.
        """
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_length = pad_to_length
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples):
        return collate_tokens_values(
            samples,
            self.pad_idx,
            left_pad=self.left_pad,
            pad_to_length=self.pad_to_length,
            pad_to_multiple=self.pad_to_multiple
        )