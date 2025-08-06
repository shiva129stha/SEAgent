# An environment with AgentRewardBench, vllm installed.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Zery/CUA_World_State_Model" --tensor-parallel-size 4 --limit-mm-per-prompt image=5 --port 8001 --max-num-seqs 4 > /dev/null 2>&1 &
sleep 2m
OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://127.0.0.1:8001/v1/ python scripts/run_judge.py --frame_num 5 \
    --judge qwen-2.5-vl-screen --model_size 7B --model_path Zery/CUA_World_State_Model --base_dir trajectories/cleaned --base_save_dir judgments_new_7b_web_dist_caption_5
