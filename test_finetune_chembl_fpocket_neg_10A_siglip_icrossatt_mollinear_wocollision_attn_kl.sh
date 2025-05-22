source ~/miniconda3/etc/profile.d/conda.sh
conda activate unimol
if [[ -z "$1" ]]; then
    device="0"
else 
    device=$1
fi

if [[ -z "$2" ]]; then
    TASK="DUDE"
else
    TASK=$2
fi
batch_size=32
weight_path="savedir/finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl/2025-05-15_02-10-52/checkpoint_best.pt"

# Extract the base directory path without the filename
base_dir="${weight_path%/*}"  # Remove everything after the last "/"

# Construct results_path by replacing "savedir" with "./test"
results_path="./test_ft/${base_dir#savedir/}"  # Replace "savedir" with "./test"

echo "Results path: $results_path"


CUDA_VISIBLE_DEVICES=$device python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip_adaptor  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \
       --add-mol-linear \
       --adaptor-type identical_cross_attention \
       --test-ensemble adapt