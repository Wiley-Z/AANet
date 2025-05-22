# AANet

## Requirements

same as [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol)

```
pip install zstandard rdkit-pypi==2022.9.3
```

or running in the docker env of unicore.

## Prepare

### Checkpoint

Download the model checkpoint and put it in `savedir/finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl` directory, same as the test script.

### Test data

Download dataset. `dataset/dude_apo` contains the 38-target subset of DUD-E dataset with holo, af2 and apo structures. `dataset/lit_pcba` contains the pocket data for 12-traget subset of PCBA dataset. Molecule datasets should be either downloaded from DrugCLIP or prepared from the original dataset.

## Run

### Training

Run `bash fpocket_neg_10A_siglip.sh` for alignment phase and `finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh` for aggregation phase. You should modify **conda environment** and some path or running in the docker env of unicore.

### Test

Trained models were saved in `./savedir` directory. Run bash shell starts with `bash test_finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh <device_id> <task_name >` and pass device_id and task_name to evaluate the model on DUD-E, PCBA and COACH420 datasets. The results will be saved in `./test` directory.
