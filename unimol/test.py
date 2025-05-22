#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore
import hashlib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve


def compute_md5(file_path):
    """Compute the MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):  # Read the file in chunks
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state_md5 = compute_md5(args.path)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    if args.test_task=="DUDE":
        if "10A" in args.path:
            task.test_dude(model, pocket_lmdbs=[
            "dataset/dude_apo/holo_prep_aligned_pocket_10A/pockets.lmdb",
            "dataset/dude_apo/af2_prep_aligned_fpocket_10A/pockets.lmdb",
            "dataset/dude_apo/apo_prep_aligned_fpocket_10A/pockets.lmdb",
            ], model_identifier=state_md5, keep_score=args.keep_score)
        else:
            task.test_dude(model, pocket_lmdbs=[
                "dataset/dude_apo/holo_prep_aligned_pocket_6A/pockets.lmdb",
                "dataset/dude_apo/af2_prep_aligned_fpocket_6A/pockets.lmdb",
                "dataset/dude_apo/apo_prep_aligned_fpocket_6A/pockets.lmdb",
            ], model_identifier=state_md5, keep_score=args.keep_score)

    elif args.test_task=="PCBA":
        if "10A" in args.path:
            task.test_pcba(model, pocket_lmdbs=[
                "dataset/lit_pcba/litpcba_pocket_10A_minor/pockets.lmdb",
                "dataset/lit_pcba/litpcba_af2_fpocket_10A_minor/pockets.lmdb",
                "dataset/lit_pcba/litpcba_apo_fpocket_10A/pockets.lmdb"
            ], model_identifier=state_md5)
        else:
            task.test_pcba(model, pocket_lmdbs=[
                "dataset/lit_pcba/litpcba_pocket_6A_minor/pockets.lmdb",
                "dataset/lit_pcba/litpcba_af2_fpocket_6A_minor/pockets.lmdb",
                "dataset/lit_pcba/litpcba_apo_fpocket_6A/pockets.lmdb",
            ], model_identifier=state_md5)
    
    elif args.test_task=="COACH420":
        if "10A" in args.path:
            task.test_dude_retrieve_fpocket(
                model, 
                pocket_lmdb="dataset/p2rank_datasets/coach420_dedup/pockets_10A.lmdb",
                mol_lmdb="dataset/p2rank_datasets/coach420_dedup/mols.lmdb",
            )
        else:
            task.test_dude_retrieve_fpocket(
                model,
                pocket_lmdb="dataset/p2rank_datasets/coach420_dedup/pockets_6A.lmdb",
                mol_lmdb="dataset/p2rank_datasets/coach420_dedup/mols.lmdb",
            )
        
    elif args.test_task=="DUDE_BLIND":
        if "10A" in args.path:
            task.test_dude_zscore(model, pocket_lmdbs=[
                "dataset/dude_apo/af2_prep_aligned_fpocket_10A_all/pockets.lmdb",
                "dataset/dude_apo/apo_prep_aligned_fpocket_10A_all/pockets.lmdb",
            ], model_identifier=state_md5)
        else:
            task.test_dude_zscore(model, pocket_lmdbs=[
                "dataset/dude_apo/af2_prep_aligned_fpocket_6A_all/pockets.lmdb",
                "dataset/dude_apo/apo_prep_aligned_fpocket_6A_all/pockets.lmdb",
            ], model_identifier=state_md5)
    elif args.test_task=="PCBA_BLIND":
        if "10A" in args.path:
            task.test_dude_zscore(model, pocket_lmdbs=[
                "dataset/lit_pcba/litpcba_af2_fpocket_10A_minor_all/pockets.lmdb",
                "dataset/lit_pcba/litpcba_apo_fpocket_10A_all/pockets.lmdb",
            ], save_dir="pcba_blind", model_identifier=state_md5)
        else:
            task.test_dude_zscore(model, pocket_lmdbs=[
                "dataset/lit_pcba/litpcba_af2_fpocket_6A_minor_all/pockets.lmdb",
                "dataset/lit_pcba/litpcba_apo_fpocket_6A_all/pockets.lmdb",
            ], save_dir="pcba_blind", model_identifier=state_md5)
            
    def diff_model_params(params1, params2):
        all_keys = set(params1.keys()).union(set(params2.keys()))

        for key in all_keys:
            if key not in params1:
                print(f"Only in model2: {key}")
            elif key not in params2:
                print(f"Only in model1: {key}")
            elif not torch.equal(params1[key], params2[key].to(torch.float32)):
                print(f"Different values: {key}")
    model.to("cpu")
    diff_model_params(state["model"], model.state_dict())

def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE", help="test task", choices=["DUDE", "PCBA", "DUDE_BLIND", "PCBA_BLIND", "COACH420"])
    parser.add_argument("--keep-score", action="store_true", help="keep individual scores")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
