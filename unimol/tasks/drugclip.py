# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from IPython import embed as debug_embedded
import logging
import os
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
import torch
import random
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset, LMDBDataset as OldLMDBDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset, RawDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord, LMDBDataset, ChoiceDataset, ListKeyCutoffDataset, KeyChoiceDataset, ListKeyCatalogChoiceDataset, MappingDataset, LMDBKeyDataset, MapItemDataset,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
logger = logging.getLogger(__name__)


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio*n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp>= num:
                break
    return (tp*n)/(p*fp)


def z_score(res_new):
    medians = np.median(res_new, axis=1, keepdims=True)
    # get mad for each row
    mads = np.median(np.abs(res_new - medians), axis=1, keepdims=True)
    # get z score
    res_new = 0.6745 * (res_new - medians) / (mads + 1e-6)
    # get max for each column
    #res_max = np.max(res_cur, axis=0)
    res_max = np.max(res_new, axis=0)
    return res_max
    

def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    #print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break
    
    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    #print(res)
    #print(res2)
    return res2

def cal_metrics(y_true, y_score, alpha=80.5):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    if isinstance(alpha, Iterable):
        bedroc = {}
        for a in alpha:
            bedroc[a] = CalcBEDROC(scores, 1, a)
    else:
        bedroc = CalcBEDROC(scores, 1, alpha)
    count = 0
    # sort y_score, return index
    index  = np.argsort(y_score)[::-1]
    for i in range(int(len(index)*0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list



@register_task("drugclip")
class DrugCLIP(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument("--fpocket_aug", action="store_true", help="whether use fpocket augmentation")
        parser.add_argument("--subset", type=str, help="subset of the dataset")
        parser.add_argument("--token_list", type=str, help="token list", default="./data")
        parser.add_argument("--fpocket_neg", action="store_true", help="whether use fpocket negative samples")
        parser.add_argument("--init_logit_scale", type=float, default=np.log(10), help="initial logit scale")
        parser.add_argument("--init_logit_bias", type=float, default=-10, help="initial logit bias")
        parser.add_argument("--test-ensemble", type=str, default=None, help="test ensemble method")

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.mol_reps = None
        self.keys = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.token_list, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.token_list, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    @staticmethod
    def PrependAndAppend(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)
    
    def wrap_pocket(self, dataset):
        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )
        dataset = NormalizeDataset(dataset, "pocket_coordinates")
        src_pocket_dataset = KeyDataset(dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(dataset, "pocket_coordinates")
        src_pocket_dataset = self.PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = self.PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )
        return src_pocket_dataset, pocket_edge_type, distance_pocket_dataset, coord_pocket_dataset, pocket_len_dataset

    def wrap_ligand(self, dataset):
        dataset = KeyChoiceDataset(dataset, ["coords"])
        dataset = RemoveHydrogenDataset(
            dataset,
            "atom_types",
            "coords",
            True,
            True,
        )
        dataset = NormalizeDataset(dataset, "coords")
        src_dataset = KeyDataset(dataset, "atom_types")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coords")
        src_dataset = self.PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = self.PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
        return src_dataset, edge_type, distance_dataset, coord_dataset, mol_len_dataset
        

    def load_dataset(self, split, **kwargs):  # TODO: add split
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = self.args.data
        lig_dataset = LMDBDataset(data_path)
        lig_dataset.set_default_split(f"lig_{split}{'_' if self.args.subset else ''}{self.args.subset}")
        logger.info(f"len({self.args.subset} of {split}): {len(lig_dataset)}")
        # TODO: add a mapping dataset to remove redundant pockets for siu
        pkt_dataset = LMDBDataset(data_path)
        pkt_dataset.set_default_split(f"true_pocket_{split}{'_' if self.args.subset else ''}{self.args.subset}")

        if "siu" in data_path:
            pkt_dataset.set_default_split("true_pocket")
            sample_id_dataset = LMDBKeyDataset(data_path)
            sample_id_dataset.set_default_split(f"lig_{split}{'_' if self.args.subset else ''}{self.args.subset}")
            pkt_dataset = MapItemDataset(pkt_dataset, sample_id_dataset, idx_func=lambda x: x.rsplit('/', 1)[0])

        src_dataset, edge_type, distance_dataset, coord_dataset, mol_len_dataset = self.wrap_ligand(lig_dataset)
        src_pocket_dataset, pocket_edge_type, distance_pocket_dataset, coord_pocket_dataset, pocket_len_dataset = self.wrap_pocket(pkt_dataset)

        
        nest_dataset = {
            "net_input": {
                "mol_src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "mol_src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "mol_src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                "mol_src_coord": RightPadDatasetCoord(
                    coord_dataset,
                    pad_idx=0,
                ),
                "pocket_src_tokens": RightPadDataset(
                    src_pocket_dataset,
                    pad_idx=self.pocket_dictionary.pad(),
                ),
                "pocket_src_distance": RightPadDataset2D(
                    distance_pocket_dataset,
                    pad_idx=0,
                ),
                "pocket_src_edge_type": RightPadDataset2D(
                    pocket_edge_type,
                    pad_idx=0,
                ),
                "pocket_src_coord": RightPadDatasetCoord(
                    coord_pocket_dataset,
                    pad_idx=0,
                ),
                "mol_len": RawArrayDataset(mol_len_dataset),
                "pocket_len": RawArrayDataset(pocket_len_dataset),
            },
        }
        
        if "siu" in data_path:
            mol_id_dataset = MappingDataset(sample_id_dataset, lambda x: x.rsplit("/", 1)[-1], new_key=None)  # inchikey
            protein_id_dataset = MappingDataset(sample_id_dataset, lambda x: x.split("/", 1)[0], new_key=None)  # uniprot
            nest_dataset.update({
                "mol_id": RawArrayDataset(mol_id_dataset),
                "protein_id": RawArrayDataset(protein_id_dataset),
            })

        if self.args.fpocket_aug is True:
            fpkt_dataset = LMDBDataset(data_path)
            fpkt_dataset.set_default_split(f"fpocket_{split}{'_' if self.args.subset else ''}{self.args.subset}")
            if "siu" in data_path:
                fpkt_dataset.set_default_split("fpocket")
                fpkt_dataset = MapItemDataset(fpkt_dataset, sample_id_dataset, idx_func=lambda x: x.rsplit('/', 1)[0] + "_fpocket")
            if self.args.fpocket_neg is True:
                fpkt_dataset = ListKeyCatalogChoiceDataset(fpkt_dataset, "iou", catalog=[(lambda x: x >= 0.5), (lambda x: x <= 0.1)])
            else:
                fpkt_dataset = ListKeyCutoffDataset(fpkt_dataset, "iou", lambda x: x >= 0.5)
            fpkt_dataset = ChoiceDataset(fpkt_dataset)
            fpkt_dataset = MappingDataset(fpkt_dataset, lambda x: x["iou"] >= 0.5, new_key="mask")
            mask_dataset = KeyDataset(fpkt_dataset, "mask")
            fpkt_dataset = KeyDataset(fpkt_dataset, "inputs")
            src_fpocket_dataset, fpocket_edge_type, distance_fpocket_dataset, coord_fpocket_dataset, fpocket_len_dataset = self.wrap_pocket(fpkt_dataset)
            nest_dataset["net_input"].update({
                "fpocket_src_tokens": RightPadDataset(
                    src_fpocket_dataset,
                    pad_idx=self.pocket_dictionary.pad(),
                ),
                "fpocket_src_distance": RightPadDataset2D(
                    distance_fpocket_dataset,
                    pad_idx=0,
                ),
                "fpocket_src_edge_type": RightPadDataset2D(
                    fpocket_edge_type,
                    pad_idx=0,
                ),
                "fpocket_src_coord": RightPadDatasetCoord(
                    coord_fpocket_dataset,
                    pad_idx=0,
                ),
                "fpocket_mask": RawArrayDataset(mask_dataset),
                "fpocket_len": RawArrayDataset(fpocket_len_dataset),
            })

        nest_dataset = NestedDictionaryDataset(nest_dataset)

        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset


    

    def load_mols_dataset(self, data_path,atoms,coords, **kwargs):
        if os.path.isfile(data_path):
            dataset = OldLMDBDataset(data_path)
        else:
            dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset
    

    def load_retrieval_mols_dataset(self, data_path,atoms,coords, **kwargs):
        if os.path.isfile(data_path):
            dataset = OldLMDBDataset(data_path)
        else:
            dataset = LMDBDataset(data_path)
            if "split" in kwargs.keys():
                dataset.set_default_split(kwargs["split"])
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        # smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                # "smi_name": RawArrayDataset(smi_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):

        if os.path.isfile(data_path):
            dataset = OldLMDBDataset(data_path)
        else:
            dataset = LMDBDataset(data_path, subdir=os.path.isdir(data_path))
            if "split" in kwargs.keys():
                dataset.set_default_split(kwargs["split"])
 
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")
        other_label_dataset = KeyDataset(dataset, "other_labels")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )




        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")



        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
                "other_labels": RawDataset(other_label_dataset),
            },
        )
        return nest_dataset

    

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
            
        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)

        logger.debug(model.logit_scale)
        return model

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output


    def test_dude_target(self, target, pocket_reps, model, dataset="dude", **kwargs):
        ensemble = kwargs.get("ensemble", "max")
        if dataset == "dude" :
            basedir = "dataset/dude/all/"
        elif dataset == "litpcba":
            basedir = "dataset/lit_pcba/"
        data_path = basedir + target + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        mol_dataset.set_epoch(0)
        bsz=256
        mol_reps = []
        mol_feat = []
        mol_names = []
        labels = []
        
        # generate mol data
        cache_path = os.path.join(self.args.results_path.replace("_zscore", ""), dataset, f'{target}_mols{kwargs.get("model_identifier", "")}.npz')
        # if os.path.exists(cache_path) and ensemble != "adapt":
        if os.path.exists(cache_path):
            arrays = np.load(cache_path)
            mol_reps, labels = arrays['mol_reps'], arrays['labels']
            if ensemble == "adapt":
                mol_feat = arrays['mol_feat']
        else:
            mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater, num_workers=self.args.num_workers)
            with torch.no_grad():
                for _, sample in enumerate(tqdm(mol_data)):
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    # mol_emb = mol_encoder_rep
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    #print(mol_emb.dtype)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    #print(mol_emb.dtype)
                    # if ensemble == 'adapt':
                    #     mol_emb2 = model.mol_project_2(mol_encoder_rep)
                    #     mol_emb2 = mol_emb2 / mol_emb2.norm(dim=-1, keepdim=True)
                    #     mol_emb2 = mol_emb2.detach().cpu().numpy()
                    mol_reps.append(mol_emb)
                    mol_feat.append(mol_encoder_rep.detach().cpu().numpy())
                    mol_names.extend(sample["smi_name"])
                    labels.extend(sample["target"].detach().cpu().numpy())
            mol_reps = np.concatenate(mol_reps, axis=0)
            mol_feat = np.concatenate(mol_feat, axis=0)
            labels = np.array(labels, dtype=np.int32)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, mol_reps=mol_reps, labels=labels, mol_feat=mol_feat)
        # generate pocket data
        
        if len(pocket_reps.shape) == 1:
            pocket_reps = pocket_reps.reshape(1, -1)
        if ensemble == 'adapt':
            with torch.no_grad():
                mol_reps_gpu = torch.from_numpy(mol_reps).cuda()
                pocket_reps_gpu = torch.from_numpy(pocket_reps).cuda()
                logger.debug(f"{mol_reps_gpu.shape}, {pocket_reps_gpu.shape}")
                pocket_reps_agg = model.adaptor.infer_one(mol_rep=mol_reps_gpu, pocket_rep=pocket_reps_gpu)
                if isinstance(pocket_reps_agg, tuple):
                    pocket_reps, attn = pocket_reps_agg
                else:
                    pocket_reps = pocket_reps_agg
                if hasattr(model.adaptor, 'mol_linear'):
                    pocket_reps, mol_reps = pocket_reps
                    mol_reps = mol_reps.squeeze(0)
                    mol_reps = mol_reps / mol_reps.norm(dim=-1, keepdim=True)
                    mol_reps = mol_reps.detach().cpu().numpy()
                pocket_reps = pocket_reps.squeeze(0)
                logger.debug(f"pocket_reps_agg shape: {pocket_reps.shape}")
                # if not hasattr(model.adaptor, 'agg_project'):
                #     pocket_reps = model.pocket_project(pocket_reps)
                pocket_reps = pocket_reps / pocket_reps.norm(dim=-1, keepdim=True)
                pocket_reps = pocket_reps.detach().cpu().numpy()
                res = (pocket_reps * mol_reps).sum(axis=-1, keepdims=True).T
                logger.debug(f"{res}, {pocket_reps_agg[-1]}")
        else:
            res = pocket_reps @ mol_reps.T
        
        if res.shape[0] != 1:
            if ensemble == "max":
                res_single = res.max(axis=0)
            elif ensemble == "mean":
                res_single = res.mean(axis=0)
            elif ensemble == "zscore":
                res_single = z_score(res)
        else:
            res_single = res.max(axis=0)

        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, kwargs.get("alpha", 80.5))
        
        
        return auc, bedroc, ef_list, re_list, res_single, labels

    def encode_pockets(self, model, pocket_lmdb, batch_size=16, return_rep=False, return_other_labels=False):
        pocket_dataset = self.load_pockets_dataset(pocket_lmdb)
        pocket_dataset.set_epoch(0)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=batch_size, collate_fn=pocket_dataset.collater, shuffle=False, num_workers=8)
        pocket_reps, pocket_names = [], []
        if return_other_labels:
            pocket_other_labels = []

        with torch.no_grad():
            for _, sample in enumerate(tqdm(pocket_data)):
                sample = unicore.utils.move_to_cuda(sample)
                if return_other_labels:
                    pocket_other_labels.extend(sample["other_labels"])
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:,0,:]
                #pocket_emb = pocket_encoder_rep
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                if return_rep:
                    pocket_emb = pocket_encoder_rep.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
                pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        if return_other_labels:
            return pocket_reps, pocket_names, pocket_other_labels
        return pocket_reps, pocket_names

    def test_dude(self, model, pocket_lmdbs=["dataset/6A_pockets_baseline/pockets.lmdb"], keep_score=False, **kwargs):
        os.makedirs(self.args.results_path + "/dude", exist_ok=True)
        metrics_file = open(self.args.results_path + "/dude/metrics.csv", "w")
        metrics_file.write("test,auc,bedroc,ef_05,ef_1,ef_2,ef_5,re_05,re_1,re_2,re_5\n")
        for pocket_lmdb in pocket_lmdbs:
            # if 'finetune' in self.args.results_path:
            #     print("use pocket feat")
            #     pocket_reps, pocket_names = self.encode_pockets(model, pocket_lmdb, return_rep=True)
            # else:
            pocket_reps, pocket_names = self.encode_pockets(model, pocket_lmdb)
            auc_list = {}
            bedroc_list = {}
            res_list= []
            labels_list = []
            re_list = {
                "0.005": {},
                "0.01": {},
                "0.02": {},
                "0.05": {},
            }
            ef_list = {
                "0.005": {},
                "0.01": {},
                "0.02": {},
                "0.05": {},
            }
            target_list = []
            for pr,target in tqdm(zip(pocket_reps, pocket_names), total=len(pocket_names)):
                target = target.split("_")[0]
                # try:
                auc, bedroc, ef, re, res_single, labels = self.test_dude_target(
                    target, pr, model, alpha=80.5, model_identifier=kwargs.get("model_identifier", ""),
                    ensemble=self.args.test_ensemble
                    )
                # except Exception as e:
                #     print(f"Error for {target} {e}")
                #     continue
                auc_list[target] = auc
                bedroc_list[target] = bedroc
                for key in ef:
                    ef_list[key][target] = ef[key]
                for key in re_list:
                    re_list[key][target] = re[key][0]
                res_list.append(res_single)
                labels_list.append(labels)
                target_list.extend([target] * len(labels))
            res = np.concatenate(res_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            print(f"dude/{pocket_lmdb.split('/')[-2]}.csv")
            print("auc mean", np.mean(list(auc_list.values())))
            print("bedroc mean", np.mean(list(bedroc_list.values())))

            for key in ef_list:
                print("ef", key, "mean", np.mean(list(ef_list[key].values())))

            for key in re_list:
                print("re", key, "mean",  np.mean(list(re_list[key].values())))
            metrics_file.write(f"{pocket_lmdb.split('/')[-2]},{np.mean(list(auc_list.values()))},{np.mean(list(bedroc_list.values()))},{np.mean(list(ef_list['0.005'].values()))},{np.mean(list(ef_list['0.01'].values()))},{np.mean(list(ef_list['0.02'].values()))},{np.mean(list(ef_list['0.05'].values()))},{np.mean(list(re_list['0.005'].values()))},{np.mean(list(re_list['0.01'].values()))},{np.mean(list(re_list['0.02'].values()))},{np.mean(list(re_list['0.05'].values()))}\n")


            # save printed results
            results = pd.DataFrame({'auc': auc_list, 'bedroc': bedroc_list, 'ef_05': ef_list["0.005"], 'ef_1': ef_list["0.01"], 'ef_2': ef_list["0.02"], 'ef_5': ef_list["0.05"], 're_05': re_list["0.005"], 're_1': re_list["0.01"], 're_2': re_list["0.02"], 're_5': re_list["0.05"]})
            results.to_csv(self.args.results_path + f"/dude/{pocket_lmdb.split('/')[-2]}.csv", index_label='target')
            if keep_score:
                print("write raw score")
                results = pd.DataFrame({'target': target_list, 'score': res.squeeze(), 'label': labels})
                results.to_csv(self.args.results_path + f"/dude/{pocket_lmdb.split('/')[-2]}_score.csv.gz", index=False)
        metrics_file.close()
        return 
    
    
    def test_dude_zscore(self, model, pocket_lmdbs=["dataset/dude/dude_fpocket_6A_all/pockets.lmdb"], save_dir="dude_zscore", **kwargs):
        os.makedirs(self.args.results_path + f"/{save_dir}", exist_ok=True)
        dataset = "dude" if "dude" in save_dir else "litpcba"
        metrics_file = open(self.args.results_path + f"/{save_dir}/metrics.csv", "w")
        metrics_file.write("test,ensemble,auc,bedroc,ef_05,ef_1,ef_2,ef_5,re_05,re_1,re_2,re_5\n")
        for pocket_lmdb in pocket_lmdbs:    
            # if 'finetune' in self.args.results_path:
            #     logger.info("use pocket feat")
            #     logger.info(f"use original pocket projector: {not hasattr(model.adaptor, 'agg_project')}")
            #     pocket_reps, pocket_names = self.encode_pockets(model, pocket_lmdb, return_rep=True)
            # else:
            pocket_reps, pocket_names = self.encode_pockets(model, pocket_lmdb)
            pocket_data = {}
            for pr, target in zip(pocket_reps, pocket_names):
                if "_" not in target:
                    # to exclude holo pockets from dude dataset, which are not used in the evaluation.
                    continue
                target, iou = target.rsplit("_", 1)
                if target not in pocket_data:
                    pocket_data[target] = []
                pocket_data[target].append((pr, float(iou)))
            for target in pocket_data:
                pocket_data[target], labels = zip(*sorted(pocket_data[target], key=lambda x: x[1], reverse=True))
                # print(target, max(labels))
                pocket_data[target] = np.stack(pocket_data[target])
            auc_list = {}
            bedroc_list = {}
            res_list= []
            labels_list = []
            re_list = {
                "0.005": {},
                "0.01": {},
                "0.02": {},
                "0.05": {},
            }
            ef_list = {
                "0.005": {},
                "0.01": {},
                "0.02": {},
                "0.05": {},
            }
            # print(pocket_data.keys())
            # raise
            # for ensemble in ["zscore", "max", "mean"]:
            ensemble = "adapt" if ("finetune" in self.args.results_path) and not ("max" in self.args.results_path) else "max"
            for ensemble in [ensemble,]:
                # if ensemble == "adapt" and not hasattr(model, "adaptor"):
                #     continue
                for target,pr in tqdm(pocket_data.items()):
                    # target = target.split("_")[0]
                    target = target.replace("-", "_")  # for pcba ESR1_ago and ESR1_ant
                    # try:
                    auc, bedroc, ef, re, res_single, labels = self.test_dude_target(
                        target, pr, model, alpha=80.5, ensemble=ensemble,
                        model_identifier=kwargs.get("model_identifier", ""), dataset=dataset
                    )
                    # except Exception as e:
                    #     print(f"Error for {target} {e}")
                    #     continue
                    auc_list[target] = auc
                    bedroc_list[target] = bedroc
                    for key in ef:
                        ef_list[key][target] = ef[key]
                    for key in re_list:
                        re_list[key][target] = re[key][0]
                    res_list.append(res_single)
                    labels_list.append(labels)
                res = np.concatenate(res_list, axis=0)
                labels = np.concatenate(labels_list, axis=0)
                print("ensemble:", ensemble)
                print(f"{save_dir}/{pocket_lmdb.split('/')[-2]}.csv")
                print("auc mean", np.mean(list(auc_list.values())))
                print("bedroc mean", np.mean(list(bedroc_list.values())))

                for key in ef_list:
                    print("ef", key, "mean", np.mean(list(ef_list[key].values())))

                for key in re_list:
                    print("re", key, "mean",  np.mean(list(re_list[key].values())))
                metrics_file.write(f"{pocket_lmdb.split('/')[-2]},{ensemble},{np.mean(list(auc_list.values()))},{np.mean(list(bedroc_list.values()))},{np.mean(list(ef_list['0.005'].values()))},{np.mean(list(ef_list['0.01'].values()))},{np.mean(list(ef_list['0.02'].values()))},{np.mean(list(ef_list['0.05'].values()))},{np.mean(list(re_list['0.005'].values()))},{np.mean(list(re_list['0.01'].values()))},{np.mean(list(re_list['0.02'].values()))},{np.mean(list(re_list['0.05'].values()))}\n")


            # save printed results
            results = pd.DataFrame({'auc': auc_list, 'bedroc': bedroc_list, 'ef_05': ef_list["0.005"], 'ef_1': ef_list["0.01"], 'ef_2': ef_list["0.02"], 'ef_5': ef_list["0.05"], 're_05': re_list["0.005"], 're_1': re_list["0.01"], 're_2': re_list["0.02"], 're_5': re_list["0.05"]})
            results.to_csv(self.args.results_path + f"/{save_dir}/{pocket_lmdb.split('/')[-2]}.csv", index_label='target')
        metrics_file.close()
        return 
    

    def test_pcba(self, model, pocket_lmdbs, **kwargs):
        os.makedirs(self.args.results_path + "/litpcba", exist_ok=True)
        metrics_file = open(self.args.results_path + "/litpcba/metrics.csv", "w")
        metrics_file.write("test,ensemble,auc,bedroc,ef_05,ef_1,ef_2,ef_5,re_05,re_1,re_2,re_5\n")
        for pocket_lmdb in pocket_lmdbs:    
            pocket_reps, pocket_names = self.encode_pockets(model, pocket_lmdb, batch_size=4)
            # print(pocket_names)
            pocket_data = {}
            for pr, pn in zip(pocket_reps, pocket_names):
                target, pdb = pn.rsplit("_", 1)
                if target not in pocket_data:
                    pocket_data[target] = []
                pocket_data[target].append(pr)
            for target in pocket_data:
                pocket_data[target] = np.stack(pocket_data[target])
            auc_list = {}
            bedroc_list = {}
            res_list= []
            labels_list = []
            re_list = {
                "0.005": {},
                "0.01": {},
                "0.02": {},
                "0.05": {},
            }
            ef_list = {
                "0.005": {},
                "0.01": {},
                "0.02": {},
                "0.05": {},
            }
            # print(pocket_data.keys())
            # raise
            # for ensemble in ["zscore", "max", "mean"]:
            for ensemble in ["max", "zscore"]:
                for target,pr in tqdm(pocket_data.items()):
                    # target = target.split("_")[0]
                    try:
                        auc, bedroc, ef, re, res_single, labels = self.test_dude_target(target, pr, model, alpha=80.5, ensemble=ensemble, dataset="litpcba", model_identifier=kwargs.get("model_identifier", ""))
                    except Exception as e:
                        print(f"Error for {target} {e}")
                        continue
                    auc_list[target] = auc
                    bedroc_list[target] = bedroc
                    for key in ef:
                        ef_list[key][target] = ef[key]
                    for key in re_list:
                        re_list[key][target] = re[key][0]
                    res_list.append(res_single)
                    labels_list.append(labels)
                res = np.concatenate(res_list, axis=0)
                labels = np.concatenate(labels_list, axis=0)
                print("ensemble:", ensemble)
                print(f"litpcba/{pocket_lmdb.split('/')[-2]}.csv")
                print("auc mean", np.mean(list(auc_list.values())))
                print("bedroc mean", np.mean(list(bedroc_list.values())))

                for key in ef_list:
                    print("ef", key, "mean", np.mean(list(ef_list[key].values())))

                for key in re_list:
                    print("re", key, "mean",  np.mean(list(re_list[key].values())))
                metrics_file.write(f"{pocket_lmdb.split('/')[-2]},{ensemble},{np.mean(list(auc_list.values()))},{np.mean(list(bedroc_list.values()))},{np.mean(list(ef_list['0.005'].values()))},{np.mean(list(ef_list['0.01'].values()))},{np.mean(list(ef_list['0.02'].values()))},{np.mean(list(ef_list['0.05'].values()))},{np.mean(list(re_list['0.005'].values()))},{np.mean(list(re_list['0.01'].values()))},{np.mean(list(re_list['0.02'].values()))},{np.mean(list(re_list['0.05'].values()))}\n")


            # save printed results
            results = pd.DataFrame({'auc': auc_list, 'bedroc': bedroc_list, 'ef_05': ef_list["0.005"], 'ef_1': ef_list["0.01"], 'ef_2': ef_list["0.02"], 'ef_5': ef_list["0.05"], 're_05': re_list["0.005"], 're_1': re_list["0.01"], 're_2': re_list["0.02"], 're_5': re_list["0.05"]})
            results.to_csv(self.args.results_path + f"/litpcba/{pocket_lmdb.split('/')[-2]}.csv", index_label='target')
        metrics_file.close()
        return 

    
    def test_dude_retrieve_fpocket(self, model, pocket_lmdb="dataset/dude/dude_fpocket_6A_all/pockets.lmdb", mol_lmdb="dataset/dude/crystal_ligand/mols.lmdb", **kwargs):
        mol_dataset = self.load_mols_dataset(mol_lmdb, "atoms", "coordinates")
        bsz=128
        mol_reps = []
        mol_names = []
        labels = []
        

        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater, num_workers=self.args.num_workers)
        with torch.no_grad():
            for _, sample in enumerate(tqdm(mol_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]
                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:,0,:]
                mol_emb = mol_encoder_rep
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                #print(mol_emb.dtype)
                mol_emb = mol_emb.detach().cpu().numpy()
                #print(mol_emb.dtype)
                mol_reps.append(mol_emb)
                mol_names.extend(sample["smi_name"])
                labels.extend(sample["target"].detach().cpu().numpy())
            mol_reps = np.concatenate(mol_reps, axis=0)
            labels = np.array(labels, dtype=np.int32)
            # generate pocket data
            pocket_reps, pocket_names, other_labels = self.encode_pockets(model, pocket_lmdb, return_other_labels=True)
        with open(os.path.join(self.args.results_path, f"{pocket_lmdb.split('/')[-2]}_pocket.pkl"), "wb") as f:
            pickle.dump([pocket_reps, pocket_names], f)
        with open(os.path.join(self.args.results_path, f"{pocket_lmdb.split('/')[-2]}_mol.pkl"), "wb") as f:
            pickle.dump([mol_reps, mol_names], f)
        
        pocket_data = {}
        for pr, target, ol in zip(pocket_reps, pocket_names, other_labels):
            if "_" not in target:
                continue
            target, iou = target.split("_")
            if target not in pocket_data:
                pocket_data[target] = []
            pocket_data[target].append((pr, float(iou), ol))
        fpocket_iou = {}
        fpocket_dca_atm_best = {}
        fpocket_dca_atm_noh_best = {}
        fcentre_dca_best = {}
        fpocket_dca_atm_rand = {}
        fpocket_dca_atm_noh_rand = {}
        fcentre_dca_rand = {}
        other_labels_sorted = {}
        for target in pocket_data:
            pocket_data[target], labels, other_labels_sorted[target] = zip(*sorted(pocket_data[target], key=lambda x: x[1], reverse=True))
            fpocket_iou[target] = labels[0]
            if other_labels_sorted[target][0] is not None:
                fpocket_dca_atm_best[target] = other_labels_sorted[target][0]["fpocket_atm_dca"]
                fpocket_dca_atm_noh_best[target] = other_labels_sorted[target][0]["fpocket_atm_dca_noh"]
                fcentre_dca_best[target] = other_labels_sorted[target][0]["fcentre_dca"]
                rand_label = random.choice(other_labels_sorted[target])
                fpocket_dca_atm_rand[target] = rand_label["fpocket_atm_dca"]
                fpocket_dca_atm_noh_rand[target] = rand_label["fpocket_atm_dca_noh"]
                fcentre_dca_rand[target] = rand_label["fcentre_dca"]
            # print(target, max(labels), labels[0])

        acc = {}
        auc = {}
        fpocket_dca_atm = {}
        fpocket_dca_atm_noh = {}
        fcentre_dca = {}
        chosen_n = {}
        same_lig = {}
        for mol_rep, target in zip(mol_reps, mol_names):
            if target not in pocket_data:
                print(f"target {target} not in pocket data (maybe empty)")
                continue
            mol_rep = mol_rep.reshape(1, -1)
            pocket_reps = np.stack(pocket_data[target])
            if hasattr(model, "adaptor") and model.adaptor is not None:
                with torch.no_grad():
                    mol_reps_gpu = torch.from_numpy(mol_rep).cuda()
                    pocket_reps_gpu = torch.from_numpy(pocket_reps).cuda()
                    logger.debug(f"{mol_reps_gpu.shape}, {pocket_reps_gpu.shape}")
                    pocket_reps_agg, attn_weights = model.adaptor.infer_one(mol_rep=mol_reps_gpu, pocket_rep=pocket_reps_gpu)
                    res = attn_weights.cpu().numpy().flatten()
            else:
                res = (mol_rep @ pocket_reps.T).flatten()
            chosen_id = res.argmax()
            rank = np.argsort(res)[::-1]
            chosen_n[target] = 1 if "n_lig" not in other_labels_sorted[target][0] else other_labels_sorted[target][0]["n_lig"]
            same_lig[target] = other_labels_sorted[target][0].get("same_lig", 1)
            if same_lig[target] != 1:
                print(f"{target} has {same_lig[target]} same ligands")
            acc[target] = (chosen_id == 0)
            auc[target] = roc_auc_score([1] + [0] * (len(pocket_reps) - 1), res)
            other_labels_sorted[target] = [other_labels_sorted[target][i] for i in rank]
            fpocket_dca_atm[target], fpocket_dca_atm_noh[target], fcentre_dca[target] = zip(
                *[(i["fpocket_atm_dca"], i["fpocket_atm_dca_noh"], i["fcentre_dca"]) for i in other_labels_sorted[target]]
            )
        df = pd.DataFrame({
            'acc': acc, 'auc': auc, 'iou': fpocket_iou,
            'fpocket_dca_atm': fpocket_dca_atm, 'fpocket_dca_atm_noh': fpocket_dca_atm_noh, 'fcentre_dca': fcentre_dca,
            'fpocket_dca_atm_best': fpocket_dca_atm_best, 'fpocket_dca_atm_noh_best': fpocket_dca_atm_noh_best, 'fcentre_dca_best': fcentre_dca_best,
            'fpocket_dca_atm_rand': fpocket_dca_atm_rand, 'fpocket_dca_atm_noh_rand': fpocket_dca_atm_noh_rand, 'fcentre_dca_rand': fcentre_dca_rand,
            'chosen_n': chosen_n,
            'same_lig': same_lig
        })
        print("acc", np.mean(list(acc.values())))
        for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"acc@{i}:", np.mean(list(df[df['iou'] >= i]['acc'].values)))
        print("auc", np.mean(list(auc.values())))
        # dca_threshold = 4
        # for i in ['fpocket_dca_atm', 'fpocket_dca_atm_noh', 'fcentre_dca']:
        #     print(f"top1_{i}", df[df[i].apply(lambda x: x <= dca_threshold)].shape[0] / df.shape[0],
        #           "for ref, [rand, best] is : [", df[df[i+"_rand"].apply(lambda x: x <= dca_threshold)].shape[0] / df.shape[0],
        #           df[df[i+"_best"].apply(lambda x: x <= dca_threshold)].shape[0] / df.shape[0], "]")
        print(df)
        for dca_threshold in range(1, 10):
            print(f"top1_fpocket_dca{dca_threshold}_atm_noh", df[["fpocket_dca_atm_noh", "same_lig"]].apply(lambda x: min(x["fpocket_dca_atm_noh"][:x["same_lig"]]) <= dca_threshold, axis=1).sum() / df.shape[0],
                  "for ref, [rand, best] is : [", df[df["fpocket_dca_atm_noh_rand"].apply(lambda x: x <= dca_threshold)].shape[0] / df.shape[0],
                  df[df["fpocket_dca_atm_noh_best"].apply(lambda x: x <= dca_threshold)].shape[0] / df.shape[0], "]")
        for dca_threshold in range(1, 10):
            print(f"topn_fpocket_dca{dca_threshold}_atm_noh", 
                  df[["fpocket_dca_atm_noh", "chosen_n"]].apply(lambda x: min(x["fpocket_dca_atm_noh"][:x["chosen_n"]]) <= dca_threshold, axis=1).sum() / df.shape[0])
        return 
        
    def encode_mols(self, model, data_path, atoms="atoms", coords="coordinates", bsz=256,**kwargs):
        mol_dataset = self.load_retrieval_mols_dataset(data_path,atoms,coords, **kwargs)
        mol_reps = []
        mol_names = []
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater, num_workers=self.args.num_workers)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            # mol_names.extend(sample["smi_name"])

        mol_reps = np.concatenate(mol_reps, axis=0)
        
        return mol_reps
    
    def encode_mols_once(self, model, data_path, emb_dir, atoms="atoms", coords="coordinates", **kwargs):
        
        # cache path is embdir/data_path.pkl

        cache_path = os.path.join(emb_dir, data_path.split("/")[-1] + ".pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                mol_reps, mol_names = pickle.load(f)
            return mol_reps, mol_names

        mol_reps = self.encode_mols(model, data_path, atoms=atoms, coords=coords)

        # save the results
        
        # with open(cache_path, "wb") as f:
        #     pickle.dump([mol_reps, mol_names], f)
            
        np.save(os.path.join(emb_dir, data_path.split("/")[-1] + ".npy"), mol_reps)

        return mol_reps, mol_names
    
    def retrieve_mols(self, model, mol_path, pocket_path, emb_dir, k, **kwargs):
 
        os.makedirs(emb_dir, exist_ok=True)        
        mol_reps, mol_names = self.encode_mols_once(model, mol_path, emb_dir,  "atoms", "coordinates")
        
        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        
        res = pocket_reps @ mol_reps.T
        res = res.max(axis=0)


        # get top k results

        
        top_k = np.argsort(res)[::-1][:k]

        # return names and scores
        
        return [mol_names[i] for i in top_k], res[top_k]


        

        
         


    

    

        
            
         

        
    
    
