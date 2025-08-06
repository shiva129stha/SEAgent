export VLLM_BASE_URL=http://YOUR.IP.ADDRESS:PORT/v1
model_name="UI-TARS-1.5-7B"
timeout 6h python run_multienv_uitars_1_5.py \
    --headless --observation_type screenshot_a11y_tree --model ui_tars_1.5_7b \
    --result_dir ./results_en_test_1.5_fullosworld/all_l15_h5/${model_name} --num_envs 8 --sleep_after_execution 0.5 \
    --max_tokens 1000 --top_p 0.9 --temperature 1.0 --max_trajectory_length 15 --history_n 5

model_name="UI-TARS-1.5-7B"
timeout 8h python run_multienv_uitars_1_5.py \
    --headless --observation_type screenshot_a11y_tree --model ui_tars_1.5_7b \
    --result_dir ./results_en_test_1.5_fullosworld/all_l50_h5/${model_name} --num_envs 8 --sleep_after_execution 0.5 \
    --max_tokens 1000 --top_p 0.9 --temperature 1.0 --max_trajectory_length 50 --history_n 5
