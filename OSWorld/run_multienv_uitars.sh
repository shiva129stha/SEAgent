model_name="UI-TARS-7B-DPO"

timeout 8h python run_multienv_uitars.py \
    --headless --observation_type screenshot_a11y_tree --model ui_tars_7b_dpo \
    --result_dir ./results_en_test_fullosworld/all_l15_h1/${model_name} --num_envs 8 --sleep_after_execution 0.5 \
    --max_tokens 1000 --top_p 0.9 --temperature 1.0 --max_trajectory_length 15 --history_n 1
