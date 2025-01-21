#!/bin/bash
#SBATCH --partition=gpu_bwanggroup
#SBATCH --account=bwanggroup_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=echomamba-pretrain
#SBATCH --output=/cluster/home/t115318uhn/VideoMamba/videomamba/video_sm/pretrain-%j.out

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_middle_mask_echo_pt_f8_res384'
OUTPUT_DIR="/cluster/home/t115318uhn/VideoMamba/videomamba/video_sm/outputs/${JOB_NAME}"
LOG_DIR="/cluster/home/t115318uhn/VideoMamba/videomamba/video_sm/logs/${JOB_NAME}"
PREFIX='/cluster/projects/bwanggroup/echo_reports/data'
DATA_PATH='/cluster/home/t115318uhn/VideoMamba/videomamba/data/videomamba_uhn_echo.csv'

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Modified parameters to match your SBATCH settings
GPUS=4                    # Total number of GPUs
GPUS_PER_NODE=4          # GPUs per node
CPUS_PER_TASK=8          # CPUs per task from SBATCH

srun \
    --job-name=${JOB_NAME} \
    --gres=gpu:a100:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    python -u run_videomamba_pretraining.py \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --num_sample 1 \
    --split ',' \
    --flip True \
    --mask_type 'attention' \
    --mask_ratio 0.8 \
    --model 'videomamba_middle_pretrain' \
    --clip_teacher 'clip_b16' \
    --clip_loss_ratio 1 \
    --clip_loss_type 'l2' \
    --clip_decoder_embed_dim 576 \
    --clip_output_dim 512 \
    --clip_norm_type 'l2' \
    --clip_return_layer 1 \
    --clip_return_interval 1 \
    --clip_student_return_interval 1 \
    --clip_return_cls \
    --clip_return_attn \
    --tubelet_size 1 \
    --lr 1.5e-4 \
    --drop_path 0.4 \
    --batch_size 296 \
    --num_segments 8 \
    --num_frames 8 \
    --sampling_rate 1 \
    --num_workers 4 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_steps 100 \
    --save_ckpt_freq 1000 \
    --epochs 801 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --bf16
