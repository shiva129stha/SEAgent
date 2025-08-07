# SEAgent

This repository is the official implementation of SEAgent.

**[SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience](https://arxiv.org/abs/2508.04700)**
</br>
[Zeyi Sun](https://sunzey.github.io/),
[Ziyu Liu](https://liuziyu77.github.io/),
[Yuhang Zang](https://yuhangzang.github.io/),
[Yuhang Cao](https://scholar.google.com/citations?user=sJkqsqkAAAAJ/),
[Xiaoyi Dong](https://lightdxy.github.io/),
[Tong Wu](https://wutong16.github.io/),
[Dahua Lin](http://dahua.site/),
[Jiaqi Wang](https://myownskyw7.github.io/)
<!-- <p align="center">
<a href="https://arxiv.org/abs/2312.03818"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p> -->
<p align="center">
📖<a href="https://arxiv.org/abs/2508.04700">Paper</a> |
🤗<a href="https://huggingface.co/Zery/SEAgent-1.0-7B">SEAgent-1.0-7B</a> | 🤗<a href="https://huggingface.co/Zery/CUA_World_State_Model">World State Model-7B</a></h3>
</p>

## 👨‍💻 Todo
- [ ] Training code of SEAgent based on OpenRLHF.
- [x] Training code of SEAgent based on R1-V.
- [x] Task Generation code based on Curriculum Generator.
- [x] Inference code of SEAgent on OSWorld.
- [x] Inference code of World-State-Model on AgentRewardBench.
- [x] Release of SEAgent-1.0-7B.
- [x] Release of World-State-Model-1.0-7B.

## 🛠️ Usage
### Installation
```shell
conda create -n seagent python=3.11 
conda activate seagent
bash setup.sh
```


## Training
RL: `src/r1-v/run_grpo_gui_8_7b.sh`

SFT: `sft.sh`

## Inference
```shell
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
    --max_tokens 1000 --top_p 0.9 --temperature 1.0 --max_trajectory_length 15 --history_n 1

```

### OSworld
```shell
cd OSWorld
```
Change IP to vllm deployed UI_TARS or SEAgent.
```shell
bash run_multienv_uitars_1_0_full.sh
```

## Sample Curriculum Data.

### Generate Task Instructions
```shell
cd OSWorld
```
Deploy World State Model and Curriculum Generator for task generation.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Qwen/Qwen2.5-72B-Instruct" --served-model-name qwen72b --port 8002 --tensor-parallel-size 4
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "Zery/CUA_World_State_Model" --served-model-name gui_judge --port 8001 --tensor-parallel-size 4
```
After deploy model for task generation, use `task_buffer/task_buffer_update_from_qwen.py` to update task. 

```shell
python task_buffer/task_buffer_update_from_qwen.py \
    --judge_model gui_judge \
    --phase -1 \
    --software vscode \
    --base_result_dir results_en_phase_qwen
```
Set different software and phase number (-1 for initial phase) to generate curriculumed tasks.

### Executing Instructions
Deploy Actor Model (UI-TARS as initialization.)

```shell
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --served-model-name ui-tars --port 8001 \
    --model bytedance-research/UI-TARS-7B-DPO --limit-mm-per-prompt image=5 -tp 1
```
run `run_multienv_uitars_new_traj_evolve.sh` to sample trajectories with actor agent.

```shell
export VLLM_BASE_URL=http://YOUR.IP.ADDRESS:PORT/v1
for software in vs_code; do
    python run_multienv_uitars_new_traj_evolve.py \
        --headless --observation_type screenshot_a11y_tree --model ui_tars_7b_dpo --test_all_meta_path ./task_buffer/task_buffer_qwen_${software}_phase0.json \
        --result_dir ./results_en_phase0/7b_new_traj_multi_env=8_${software}_new_fix_nimg=1_maxtraj=15_t=0.0_r0 --num_envs 8 \
        --max_tokens 1000 --top_p 0.9 --temperature 1.0 --max_trajectory_length 15 --history_n 1 --software ${software}
done
```

### Judge the trajectories and generate pseudo labeled SFT/RL data with World State Model.
1. Deploy World State Model via vllm.
```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "Zery/CUA_World_State_Model" --served-model-name gui_judge --port 8001 --tensor-parallel-size 4
```
2. Use `judge_full_process.sh` to judge previous generated trajectories.
3. Use `gen_sft.py` to generate training data for training process. The `7b_positive_xxx.json` can be directly used for SFT (Behavior Cloning) training. The `7b_positive_xxx.json` and `7b_negative_xxx.json` can be used for RL. 
You can use `visualize_data.py` to visualize generated data.

## Evaluation of World State Model on AgentRewardBench.
We test our World State Model on [AgentRewardBench](https://github.com/McGill-NLP/agent-reward-bench) to test its accuracy for judging the success/failure for agent's trajectories. As it adopts middle states' screenshots for judgment, you need to modify some of the code to input more images with new prompt template. After install AgentRewardBench, replace the `scripts/run_judge.py`, `agent_reward_bench/judge/__init__.py`, `agent_reward_bench/judge/defaults.py` with newly added functions and prompt template.

Use `bash run_judge.sh` to reproduce our results.

## Evaluation of World State Model on OSWorld.
use `OSWorld/eval_osworld_bench.py`

## Acknowledgements
We sincerely thank [UI-TARS](https://github.com/bytedance/UI-TARS), [OSWorld](https://github.com/xlang-ai/OSWorld), [R1-V](https://github.com/Deep-Agent/R1-V), [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), for providing open source resources and to build the project.

## ✒️ Citation
```
@misc{sun2025seagentselfevolvingcomputeruse,
      title={SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience}, 
      author={Zeyi Sun and Ziyu Liu and Yuhang Zang and Yuhang Cao and Xiaoyi Dong and Tong Wu and Dahua Lin and Jiaqi Wang},
      year={2025},
      eprint={2508.04700},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.04700}, 
}
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## Acknowledgement
We sincerely thank projects <a href="https://github.com/Deep-Agent/R1-V">UI-TARS</a>, <a href="https://os-world.github.io/">OSWorld</a>, <a href="https://github.com/McGill-NLP/agent-reward-bench">AgentRewardBench</a>, <a href="https://github.com/Deep-Agent/R1-V">R1-V</a>, for providing their open-source resources.