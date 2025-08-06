# deploy SEAgent-1.0-7B model

vllm serve Zery/SEAgent-1.0-7B \
    --served-model-name "ui-tars-1.0-7b" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" &

export UI_TARS_1_0_URL=http://YOUR.IP.ADDRESS:PORT/v1
model_name="ui-tars-1.0-7b"
# test on five software only.
python run_multienv_uitars_1_0.py \
    --headless --observation_type screenshot --model ui-tars-1.0-7b \
    --result_dir ./results_en_test_1_0/all_l15_h5/${model_name} --num_envs 8 --sleep_after_execution 2.0 \
    --max_tokens 1000 --top_p 0.9 --temperature 0.0 --max_trajectory_length 15 --history_n 1
