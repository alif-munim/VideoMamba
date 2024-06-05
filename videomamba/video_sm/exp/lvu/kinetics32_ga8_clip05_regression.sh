export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='clip_logdir/kinetics32_clip05_bs2_ga8'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/cluster/projects/bwanggroup/echo_reports/data/studies'
DATA_PATH='/cluster/home/t115318uhn/VideoMamba/videomamba/ef_log_norm'

VIDEO_RESOLUTION=384
REPEAT_SAMPLE=1
SAVE_FREQ=250

VIDEO_LEN=32
BATCH_SIZE=2 # Change depending on VRAM
GRAD_ACCUMULATION_STEPS=8

python run_regression_finetuning.py \
    --model videomamba_middle \
    --finetune '/cluster/home/t115318uhn/VideoMamba/videomamba/video_sm/models/videomamba_m16_k400_mask_pt_f8_res224.pth' \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'ECHO' \
    --split ',' \
    --nb_classes 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_sample ${REPEAT_SAMPLE} \
    --input_size ${VIDEO_RESOLUTION} \
    --short_side_size ${VIDEO_RESOLUTION} \
    --num_frames ${VIDEO_LEN} \
    --orig_t_size ${VIDEO_LEN} \
    --num_workers 8 \
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
    --log_mae \
    --clip_grad 0.5 \
    --save_freq ${SAVE_FREQ} \
    --grad_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
    --update_freq ${GRAD_ACCUMULATION_STEPS} \
