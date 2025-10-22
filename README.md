# AANet

## Requirements

same as [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol)

```
pip install zstandard rdkit-pypi==2022.9.3
```

or running in the docker env of [unicore](https://hub.docker.com/layers/dptechnology/unicore/0.0.1-pytorch1.11.0-cuda11.3/images/sha256-6210fae21cdf424f10ba51a118a8bcda90d927822e1fea0070f87cb4d5f2a6d2).

## Prepare

### Checkpoint

Download the model checkpoint and put it in `savedir/finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl` directory, same as the test script.

### Test data

`dataset/dude_apo` contains the 38-target subset of DUD-E dataset with holo, af2 and apo structures. `dataset/lit_pcba` contains the pocket data for 12-traget subset of PCBA dataset. Molecule datasets should be either downloaded from [DrugCLIP](https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV) and decompressed into `dataset/`, or prepared from the original dataset.

## Run

### Training

Run `bash fpocket_neg_10A_siglip.sh` for alignment phase and `finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh` for aggregation phase. You should modify **conda environment** and some path or running in the docker env of unicore.

### Test

Download the trained model from [google drive](https://drive.google.com/file/d/11Hixd7vVKg6RZcZ81LEXKWoc68p61csV) and save it in `./savedir` directory. Run bash shell starts with `bash test_finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh <device_id> <task_name >` and pass device_id and task_name to evaluate the model on DUD-E, PCBA and COACH420 datasets. The results will be saved in `./test` directory.
