"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import List, Dict
import math
from tqdm import tqdm
from multiprocessing import Process, Manager
import lib_run_single
from desktop_env.desktop_env import DesktopEnv

from mm_agents.uitars_agent import UITARSAgent
import traceback
# import wandb


#  Logger Configs {{{ #
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--software", type=str, default="vscode", help="software name"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot_a11y_tree",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--history_n", type=int, default=5, help="Number of images used.")
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    
    args = parser.parse_args()
    return args

def distribute_tasks(test_all_meta: dict, num_envs: int) -> List[Dict]:
    """Distribute tasks evenly across environments, and assign each task a unique ID."""
    # 获取所有任务列表
    all_tasks = test_all_meta.get('exam_new', [])
    
    # 包装每个任务，加上 task_id 字段
    all_tasks_with_id = [{'task_id': i, 'data': task} for i, task in enumerate(all_tasks)]

    total_tasks = len(all_tasks_with_id)
    chunk_size = (total_tasks + num_envs - 1) // num_envs  # 向上取整分配

    # 分块分配
    distributed_tasks = []
    for i in range(num_envs):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_tasks)
        chunk = all_tasks_with_id[start_idx:end_idx]
        distributed_tasks.append({'exam_new': chunk})
    return distributed_tasks

def distribute_tasks_multi_init(test_all_meta: str, num_envs: int) -> List[Dict]:
    """Distribute tasks evenly across environments, and assign each task a unique ID."""
    all_tasks_with_id = []
    for json_file in os.listdir(test_all_meta):
        test_single_init = json.load(open(os.path.join(test_all_meta, json_file), 'r'))['exam']
        init_id = json_file.split('_')[0]  # 获取初始配置的ID
        all_tasks_with_id += [{'task_id': i, 'data': task, 'init_id': init_id} for i, task in enumerate(test_single_init)]

    total_tasks = len(all_tasks_with_id)
    chunk_size = (total_tasks + num_envs - 1) // num_envs  # 向上取整分配
    # random shuffle total_tasks
    import random
    random.shuffle(all_tasks_with_id)
    # 分块分配
    distributed_tasks = []
    for i in range(num_envs):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_tasks)
        chunk = all_tasks_with_id[start_idx:end_idx]
        distributed_tasks.append({'exam_new': chunk})
    return distributed_tasks

software_initial_config = {
    'vscode': "examples/vs_code/4e60007a-f5be-4bfc-9723-c39affa0a6d3.json",
    'gimp': "examples/gimp/2e6f678f-472d-4c55-99cc-8e7c5c402a71.json",
    'libreoffice_writer': "examples/libreoffice_writer/adf5e2c3-64c7-4644-b7b6-d2f0167927e7.json",
    'libreoffice_impress': 'examples/libreoffice_impress/bf4e9888-f10f-47af-8dba-76413038b73c.json',
    'vlc': 'examples/vlc/5ac2891a-eacd-4954-b339-98abba077adb'
}

def run_env_tasks(env_idx: int, env: DesktopEnv, agent: UITARSAgent, env_tasks: dict, args: argparse.Namespace, shared_scores: list):
    """Run tasks for a single environment."""
    logger.info(f"Executing tasks in environment {env_idx + 1}/{args.num_envs}")
    domain = args.domain
    for task in env_tasks['exam_new']:           
        res_id = task['task_id']
        instruction = task['data']
        config_file = os.path.join(
            args.test_config_base_dir, f"examples/{domain}/{task['init_id']}.json"
        )
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)
        logger.info(f"[Env {env_idx+1}][Example ID]: {res_id}")
        logger.info(f"[Env {env_idx+1}][Instruction]: {example['instruction']}")
        
        example_result_dir = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            args.model,
            domain,
            task['init_id'],
            str(res_id),
        )

        os.makedirs(example_result_dir, exist_ok=True)

        try:
            lib_run_single.run_single_example(
                agent,
                env,
                example,
                args.max_steps,
                instruction,
                args,
                example_result_dir,
                shared_scores,
            )
        except Exception as e:
            logger.error(f"Exception in Env{env_idx+1} {domain}/{res_id}: {e}")
            logger.error(traceback.format_exc())
            env.controller.end_recording(
                os.path.join(example_result_dir, "recording.mp4")
            )
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(
                    json.dumps(
                        {"Error": f"Time limit exceeded in {domain}/{res_id}"},
                    )
                )
                f.write("\n")
                f.write(traceback.format_exc())
            
    env.close()


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    logger.info("Args: %s", args)
    # distributed_tasks = distribute_tasks(test_all_meta, args.num_envs)
    distributed_tasks = distribute_tasks_multi_init(test_all_meta, args.num_envs)

    # First, set up all environments
    logger.info("Setting up all environments...")
    envs = []
    agents = []
    
    for env_idx in range(args.num_envs):
        logger.info(f"Setting up environment {env_idx + 1}/{args.num_envs}")
        
        agent = UITARSAgent(
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            observation_type=args.observation_type,
            max_trajectory_length=args.max_trajectory_length,
            history_n=args.history_n
        )
        agents.append(agent)

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=agent.action_space,
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type
            in ["a11y_tree", "screenshot_a11y_tree", "som"],
        )
        envs.append(env)
    
    logger.info("All environments are ready. Starting parallel task execution...")

    # Create a shared list for scores across processes
    with Manager() as manager:
        shared_scores = manager.list()
        
        # Create and start processes for each environment
        processes = []
        for env_idx, (env, agent, env_tasks) in enumerate(zip(envs, agents, distributed_tasks)):
            p = Process(
                target=run_env_tasks,
                args=(env_idx, env, agent, env_tasks, args, shared_scores)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Convert shared list to regular list
        scores = list(shared_scores)
    
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    test(args, args.test_all_meta_path)
