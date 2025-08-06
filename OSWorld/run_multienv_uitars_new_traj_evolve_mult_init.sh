
software=libreoffice_writer
# bash start_vms_parallel.sh
# echo "sleep 10"
# sleep 10
export UI_TARS_1_0_URL="http://your.deployed.agent.model:port/v1"
# for software in libreoffice_impress libreoffice_writer vlc; do
for r in {3..8}; do
    bash start_vms_parallel.sh
    echo "sleep 10"
    sleep 10
    timeout 1h python run_multienv_uitars_new_traj_evolve_multi_init.py \
        --headless --observation_type screenshot_a11y_tree --model ui-tars --test_all_meta_path task_buffer/task_buffer_uitars_7b_qwen2.5_winstr_hi/${software} \
        --result_dir ./results_en_phase0_multi_init/7b_${software}_200_new_traj_${software}_maxtraj=15_r${r} --num_envs 16 --domain ${software} \
        --max_tokens 1000 --top_p 0.9 --temperature 1.0 --max_trajectory_length 15 --history_n 1 --sleep_after_execution 2
done
# done