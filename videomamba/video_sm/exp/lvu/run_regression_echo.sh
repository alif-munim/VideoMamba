export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_tiny_f32_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/scratch/alif/VideoMamba/full_resized_echo_data'
# DATA_PATH='/scratch/alif/VideoMamba/ef_regression' # Unnormalized EF values
DATA_PATH='/scratch/alif/VideoMamba/ef_log_norm' # Log normalization w/ zero mean, as in https://github.com/md-mohaiminul/ViS4mer/blob/main/datasets/lvu_dataset.py

python run_regression_finetuning.py \
    --model videomamba_tiny \
    --finetune '/scratch/alif/VideoMamba/videomamba/video_sm/models/videomamba_t16_k400_f8_res224.pth' \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'ECHO' \
    --split ',' \
    --nb_classes 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 32 \
    --orig_t_size 32 \
    --num_workers 12 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 70 \
    --lr 4e-4 \
    --drop_path 0.15 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best \
