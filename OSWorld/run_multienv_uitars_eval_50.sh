export UI_TARS_1_0_URL="http://101.126.156.90:11414/v1"

bash start_vms_parallel.sh
echo "sleep 10"
for r in {0..3}; do
    timeout 6h python run_multienv_uitars.py \
        --headless --observation_type screenshot_a11y_tree --model ui-tars \
        --result_dir ./results_en/7b_baseline_maxtraj=50_r${r} --num_envs 4 --sleep_after_execution 2 \
        --max_tokens 1000 --top_p 0.1 --temperature 0.0 --max_trajectory_length 50 --max_steps 50 --history_n 1 &
done
