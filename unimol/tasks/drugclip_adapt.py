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
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset, LMDBDataset as OldLMDBDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord, LMDBDataset, ChoiceDataset, ListKeyCutoffDataset, KeyChoiceDataset, ListKeyCatalogChoiceDataset,
                         MappingDataset, LMDBKeyDataset, MapItemDataset, EmbPadDataset, StackDataset,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
logger = logging.getLogger(__name__)


@register_task("adapt")
class DrugCLIPAdapt(UnicoreTask):
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
        parser.add_argument(
            "--pocket-embedding-dataset",
            type=str,
            default=None,
            help="path to pre-calculated pocket embedding dataset",
        )
        parser.add_argument(
            "--mol-embedding-dataset",
            type=str,
            default=None,
            help="path to pre-calculated mol embedding dataset",
        )
        parser.add_argument(
            "--funetune-dc-model",
            default=None,
            type=str,
            help="path to the pre-trained pocket encoder model",
        )
        parser.add_argument(
            "--frozen-mol-encoder",
            action="store_true",
            help="freeze the molecular encoder",
        )
        parser.add_argument(
            "--frozen-pocket-encoder",
            action="store_true",
            help="freeze the pocket encoder",
        )
        parser.add_argument(
            "--frozen-mol-project",
            action="store_true",
            help="freeze the molecular projector",
        )
        parser.add_argument(
            "--frozen-pocket-project",
            action="store_true",
            help="freeze the pocket projector",
        )
        parser.add_argument(
            "--weighted-adaptor",
            action = "store_true",
            help = "whether use adaptor only for weighted mean",
        )
        parser.add_argument(
            "--prot-lig-graph-path",
            type=str,
            default=None,
            help="path to the protein-ligand graph dataset",
        )
        
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

        if args.funetune_dc_model is not None:
            print("load pretrain model weight from...", args.funetune_dc_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.funetune_dc_model,
            )
            model.load_state_dict(state["model"], strict=False)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.adaptor.parameters():
            param.requires_grad = True
        if args.adaptor_type == "identical_cross_attention" and args.weighted_adaptor:
            model.adaptor.adaptor.w_vs.weight.requires_grad = False
            model.adaptor.adaptor.fc.weight.requires_grad = False
        # if args.adaptor_type == "identical_cross_attention":
        #     model.adaptor.adaptor.out_proj.weight.requires_grad = False
        #     model.adaptor.adaptor.v_proj_weight.requires_grad = False
        # for param in model.mol_project_2.parameters():
        #     param.requires_grad = True
        model.logit_scale.requires_grad = True
        model.logit_bias.requires_grad = True
        if not args.frozen_mol_encoder:
            for param in model.mol_model.parameters():
                param.requires_grad = True
        if not args.frozen_mol_project:
            for param in model.mol_project.parameters():
                param.requires_grad = True
        if not args.frozen_pocket_encoder:
            for param in model.pocket_model.parameters():
                param.requires_grad = True
        if not args.frozen_pocket_project:
            for param in model.pocket_project.parameters():
                param.requires_grad = True
        logger.info("learnable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name}")
        return model

    def load_dataset(self, split, **kwargs):  # TODO: add split
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = self.args.data
        if self.args.mol_embedding_dataset is None:
            lig_dataset = LMDBDataset(data_path)
            lig_dataset.set_default_split(f"lig_{split}_{self.args.subset}")
            src_dataset, edge_type, distance_dataset, coord_dataset, mol_len_dataset = self.wrap_ligand(lig_dataset)
        else:
            logger.info(f"load mol embedding dataset from {self.args.mol_embedding_dataset}")
            lig_dataset = LMDBDataset(self.args.mol_embedding_dataset)
            lig_dataset.set_default_split(f"lig_{split}")
        # print(lig_dataset.get_split(f"lig_valid"))
        logger.info(f"len({self.args.subset} of {split}): {len(lig_dataset)}")
        # TODO: add a mapping dataset to remove redundant pockets for siu
        pkt_dataset = LMDBDataset(self.args.pocket_embedding_dataset)
        pkt_dataset.set_default_split("pocket")
        # print(pkt_dataset.get_split("pocket"))
        graph_dataset = LMDBDataset(self.args.prot_lig_graph_path) if self.args.prot_lig_graph_path is not None else None
        graph_dataset.set_default_split("uniprot")

        if ("siu" in data_path) or ("chembl" in data_path):
            if self.args.mol_embedding_dataset is None:
                sample_id_dataset = LMDBKeyDataset(data_path)
                sample_id_dataset.set_default_split(f"lig_{split}_{self.args.subset}")
            else:
                sample_id_dataset = LMDBKeyDataset(self.args.mol_embedding_dataset)
                sample_id_dataset.set_default_split(f"lig_{split}")
            pkt_dataset = MapItemDataset(pkt_dataset, sample_id_dataset, idx_func=lambda x: x.split('/', 1)[0])
            if graph_dataset is not None:
                graph_dataset = MapItemDataset(graph_dataset, sample_id_dataset, idx_func=lambda x: x.split('/', 1)[0], non_exist_func=lambda x: set())
            pkt_dataset = ChoiceDataset(pkt_dataset)
            pkt_dataset = StackDataset(pkt_dataset)


        if self.args.mol_embedding_dataset is None:
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
                    "mol_len": RawArrayDataset(mol_len_dataset),
                },
            }
        else:
            nest_dataset = {"mol_embs": RawArrayDataset(lig_dataset)}
        print(RawArrayDataset(lig_dataset)[0])
        if "embs" in self.args.pocket_embedding_dataset:
            nest_dataset["pocket_embs"] = EmbPadDataset(pkt_dataset, pad_idx=0)
        elif "rep" in self.args.pocket_embedding_dataset:
            nest_dataset["pocket_rep"] = EmbPadDataset(pkt_dataset, pad_idx=0)
        if graph_dataset is not None:
            nest_dataset["pocket_edge"] = graph_dataset
        
        if ("siu" in data_path) or ("chembl" in data_path):
            mol_id_dataset = MappingDataset(sample_id_dataset, lambda x: x.rsplit("/", 1)[-1], new_key=None)  # inchikey
            protein_id_dataset = MappingDataset(sample_id_dataset, lambda x: x.split("/", 1)[0], new_key=None)  # uniprot
            nest_dataset.update({
                "mol_id": RawArrayDataset(mol_id_dataset),
                "protein_id": RawArrayDataset(protein_id_dataset),
            })
            if self.args.mol_embedding_dataset is not None:
                lig_source_dataset = MappingDataset(mol_id_dataset, lambda x: "_" not in x, new_key=None)
                nest_dataset["lig_source"] = RawArrayDataset(lig_source_dataset)
        print(RawArrayDataset(lig_source_dataset)[0])

        for k, v in nest_dataset.items():
            if isinstance(v, dict):
                for key, val in v.items():
                    logger.info(f"{k}.{key}: {len(val)}")
            else:
                logger.info(f"{k}: {len(v)}")
        nest_dataset = NestedDictionaryDataset(nest_dataset)

        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(lig_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset


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
