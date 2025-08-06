export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_7b.txt"
export HF_ENDPOINT=https://hf-mirror.com 
export WANDB_MODE=offline

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_gui.py \
    --output_dir out/7b_vscode_stage0 \
    --model_name_or_path bytedance-research/UI-TARS-7B-DPO \
    --dataset_positive_name /positive_data.json \
    --dataset_negative_name /negative_data.json \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 4096 \
    --max_completion_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 5 \
    --run_name UI-TARS-7B-DPO-SFT-GRPO \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance
