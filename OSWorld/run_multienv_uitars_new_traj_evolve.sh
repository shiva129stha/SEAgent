export VLLM_BASE_URL=http://YOUR.IP.ADDRESS:PORT/v1
for software in vs_code; do
    python run_multienv_uitars_new_traj_evolve.py \
        --headless --observation_type screenshot_a11y_tree --model ui_tars_7b_dpo --test_all_meta_path ./task_buffer/task_buffer_qwen_${software}_phase0.json \
        --result_dir ./results_en_phase0/7b_new_traj_multi_env=8_${software}_new_fix_nimg=1_maxtraj=15_t=0.0_r0 --num_envs 8 \
        --max_tokens 1000 --top_p 0.9 --temperature 0.0 --max_trajectory_length 15 --history_n 1 --software ${software}
done
