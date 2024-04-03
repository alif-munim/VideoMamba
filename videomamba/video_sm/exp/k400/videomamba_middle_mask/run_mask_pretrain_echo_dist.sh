export MASTER_PORT=$((22000 + $RANDOM % 20000))
export MASTER_ADDR=localhost  # Typically localhost if using a single node
export OMP_NUM_THREADS=1
export WORLD_SIZE=4  # Total number of processes (GPUs) to use for training

JOB_NAME='videomamba_middle_mask_echo_pt_f8_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/scratch/alif/VideoMamba/full_resized_echo_data'
DATA_PATH='/scratch/alif/VideoMamba/videomamba_echo_settings.csv'

# Assuming you're running this script on one machine with multiple GPUs
for (( LOCAL_RANK=0; LOCAL_RANK<$WORLD_SIZE; LOCAL_RANK++ ))
do
    export RANK=$LOCAL_RANK
    export LOCAL_RANK=$LOCAL_RANK

    # Start the process
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
        --batch_size 64 \
        --num_segments 8 \
        --num_frames 8 \
        --sampling_rate 1 \
        --num_workers 12 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 1000 \
        --epochs 801 \
        --log_dir ${LOG_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --world_size ${WORLD_SIZE} \
        --local_rank ${LOCAL_RANK} \
        --dist_url "tcp://${MASTER_ADDR}:${MASTER_PORT}" &
        # --dist_on_itp &  # Start in the background
    
    # This is crucial: delay starting the next process to ensure unique port assignment
    sleep 10
done

wait  # Wait for all processes to finish