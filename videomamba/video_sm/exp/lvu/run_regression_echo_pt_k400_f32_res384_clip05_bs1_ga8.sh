export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='clip_logdir/k400_f32_clip05_bs1_ga8_v3'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/cluster/projects/bwanggroup/echo_reports/data/studies'
DATA_PATH='/cluster/home/t115318uhn/VideoMamba/videomamba/ef_log_norm'
RESUME_PATH='/cluster/home/t115318uhn/VideoMamba/videomamba/video_sm/exp/lvu/clip_logdir/k400_f64_clip1_bs1_ga8_v2/checkpoint-latest.pth'

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
    --batch_size 1 \
    --num_sample 1 \
    --input_size 384 \
    --short_side_size 384 \
    --save_ckpt_freq 100 \
    --num_frames 32 \
    --orig_t_size 32 \
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
    --clip_grad 0.5 \
    --save_freq 250 \
    --grad_accumulation_steps 8 \
    --log_mae \
