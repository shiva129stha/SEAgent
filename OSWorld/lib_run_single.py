import datetime
import json
import logging
import os
import time
from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")

def run_seq_example(agent, env, example, max_steps, instructions, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    env.controller.start_recording()
    for instruction_index, instruction in enumerate(instructions):
        step_idx = 0
        done = False
        all_responses = []
        input_messages = []
        while not done and step_idx < max_steps:
            response, actions, parsed_responses, input_message = agent.predict(
                instruction,
                obs
            )
            all_responses.append(response)
            input_messages.append(input_message)
            for action in actions:
                # Capture the timestamp before executing the action
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                logger.info("Step %d: %s", step_idx + 1, action)
                # save image before taking action..
                with open(os.path.join(example_result_dir, f"step_{step_idx}_{action_timestamp}.png"),
                        "wb") as _f:
                    _f.write(obs['screenshot'])
                obs, reward, done, info = env.step(action, args.sleep_after_execution)

                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({
                        "step_num": step_idx + 1,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "screenshot_file_before_exec": f"step_{step_idx}_{action_timestamp}.png"
                    }))
                    f.write("\n")
                if done:
                    logger.info("The episode is done.")
                    break
            step_idx += 1
        with open(os.path.join(example_result_dir, f"step_{instruction_index}_final.png"),
                        "wb") as _f:
            _f.write(obs['screenshot'])
        with open(os.path.join(example_result_dir, f"process_{instruction_index}.txt"), 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=4, ensure_ascii=False)
        with open(os.path.join(example_result_dir, f"messages_{instruction_index}.txt"), 'w', encoding='utf-8') as f:
            json.dump(input_messages, f, indent=4, ensure_ascii=False)
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    print(example)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    all_responses = []
    input_messages = []
    while not done and step_idx < max_steps:
        response, actions, parsed_responses, input_message = agent.predict(
            instruction,
            obs
        )
        all_responses.append(response)
        input_messages.append(input_message)
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            # save image before taking action..
            with open(os.path.join(example_result_dir, f"step_{step_idx}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            # with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
            #           "wb") as _f:
            #     _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file_before_exec": f"step_{step_idx}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    with open(os.path.join(example_result_dir, f"step_{step_idx}_final.png"),
                      "wb") as _f:
        _f.write(obs['screenshot'])
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    with open(os.path.join(example_result_dir, "process.txt"), 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=4, ensure_ascii=False)
    with open(os.path.join(example_result_dir, "messages.txt"), 'w', encoding='utf-8') as f:
        json.dump(input_messages, f, indent=4, ensure_ascii=False)
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger
