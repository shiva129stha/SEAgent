# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset, load_from_disk, Dataset
from transformers import Qwen2VLForConditionalGeneration
import numpy as np
from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import json

import re
import ast
import math
from collections import Counter

reward_branch_stat = {'click': 0, 'scroll': 0, 'type': 0, 'hotkey': 0, 'drag': 0}

def ngram_counts(seq, n):
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))

def clipped_precision(candidate, reference, n):
    cand_ngrams = ngram_counts(candidate, n)
    ref_ngrams = ngram_counts(reference, n)
    overlap = {ng: min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items()}
    return sum(overlap.values()), max(1, sum(cand_ngrams.values()))

def char_bleu4(res_str, sol_str):
    candidate = list(res_str)
    reference = list(sol_str)

    weights = [0.25, 0.25, 0.25, 0.25]
    p_ns = []
    for i in range(1, 5):
        match, total = clipped_precision(candidate, reference, i)
        if total == 0:
            p_ns.append(0)
        else:
            p_ns.append(match / total)

    # 防止 log(0)
    if min(p_ns) == 0:
        return 0.0

    log_score = sum(w * math.log(p) for w, p in zip(weights, p_ns))
    bp = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference)/max(1,len(candidate)))
    return bp * math.exp(log_score)

def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""
        
        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(3)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in keys])})"
        
        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{content.strip()}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{content.strip()}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code

def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        raise ValueError(f"Action can't parse: {action_str}")


def parse_action_qwen2vl(text, factor, image_height, image_width):
    text = text.strip()
    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            raise ValueError(f"Action can't parse: {raw_str}")
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                float_numbers = [float(num) / factor for num in numbers]
                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    dataset_positive_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the positive dataset"},
    )
    dataset_negative_name: Optional[str] = field(   
        default=None,
        metadata={"help": "Path to the negative dataset"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            parsed_response = parse_action_qwen2vl(content, factor=1000, image_height=1080, image_width=1920)
            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_response,
                image_height=1080, image_width=1920,
                input_swap=False
            )
        except Exception:
            reward = 0.0
            rewards.append(reward)
            continue
        reward = 0.0
        parsed_sol = parse_action_qwen2vl(sol, factor=1000, image_height=1080, image_width=1920)
        for res_single, sol_single in zip(parsed_response, parsed_sol):
            if res_single['action_type'] == sol_single['action_type']:
                reward += 1.0
            else:
                continue
            try:
                # for different action, use different reward function
                if res_single['action_type'] in ['click', 'left_single', 'left_double', 'right_single', 'hover']:
                    res_point = np.array(json.loads(res_single['action_inputs']['start_box'])[:2])
                    sol_point = np.array(json.loads(sol_single['action_inputs']['start_box'])[:2])
                    l1_distance = np.sum(np.abs(res_point - sol_point))
                    reward += (1 - l1_distance)
                    reward_branch_stat['click'] += 1
                elif res_single['action_type'] in ["drag", "select"]:
                    res_start_box = res_single.get("start_box")
                    res_end_box = res_single.get("end_box")
                    if res_start_box and res_end_box:
                        res_x1, res_y1, res_x2, res_y2 = eval(res_start_box)  # Assuming box is in [x1, y1, x2, y2]
                        res_sx = float((res_x1 + res_x2) / 2)
                        res_sy = float((res_y1 + res_y2) / 2)
                        res_x1, res_y1, res_x2, res_y2 = eval(res_end_box)  # Assuming box is in [x1, y1, x2, y2]
                        res_ex = float((res_x1 + res_x2) / 2)
                        res_ey = float((res_y1 + res_y2) / 2)
                    sol_start_box = sol_single.get("start_box")
                    sol_end_box = sol_single.get("end_box")
                    if sol_start_box and sol_end_box:
                        sol_x1, sol_y1, sol_x2, sol_y2 = eval(sol_start_box)  # Assuming box is in [x1, y1, x2, y2]
                        sol_sx = float((sol_x1 + sol_x2) / 2)
                        sol_sy = float((sol_y1 + sol_y2) / 2)
                        sol_x1, sol_y1, sol_x2, sol_y2 = eval(sol_end_box)  # Assuming box is in [x1, y1, x2, y2]
                        sol_ex = float((sol_x1 + sol_x2) / 2)
                        sol_ey = float((sol_y1 + sol_y2) / 2)
                    # compute mIoU as reward
                    res_box = [
                            min(res_sx, res_ex), min(res_sy, res_ey),
                            max(res_sx, res_ex), max(res_sy, res_ey)
                        ]
                    sol_box = [
                        min(sol_sx, sol_ex), min(sol_sy, sol_ey),
                        max(sol_sx, sol_ex), max(sol_sy, sol_ey)
                    ]
                    # IoU
                    def compute_iou(boxA, boxB):
                        xA = max(boxA[0], boxB[0])
                        yA = max(boxA[1], boxB[1])
                        xB = min(boxA[2], boxB[2])
                        yB = min(boxA[3], boxB[3])

                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                        unionArea = boxAArea + boxBArea - interArea

                        if unionArea == 0:
                            return 0.0
                        return interArea / unionArea
                    iou_score = compute_iou(res_box, sol_box)
                    reward += iou_score
                    reward_branch_stat['drag'] += 1
                elif res_single['action_type'] == 'wait' or res_single['action_type'] == 'finished':
                    pass
                elif res_single['action_type'] in ['type']:
                    res_str = res_single['action_inputs']['content']
                    sol_str = sol_single['action_inputs']['content']
                    bleu4_score = char_bleu4(res_str, sol_str)
                    reward += bleu4_score
                    reward_branch_stat['type'] += 1
                elif res_single['action_type'] in ['hotkey']:
                    res_str = res_single['action_inputs']["key"] if 'key' in res_single['action_inputs'].keys() else res_single['action_inputs']["hotkey"]
                    sol_str = sol_single['action_inputs']["key"] if 'key' in sol_single['action_inputs'].keys() else sol_single['action_inputs']["hotkey"]
                    bleu4_score = char_bleu4(res_str, sol_str)
                    reward += bleu4_score
                    reward_branch_stat['hotkey'] += 1
                elif res_single['action_type'] == "scroll":
                    res_direction = res_single['action_inputs'].get("direction", "")
                    sol_direction = sol_single['action_inputs'].get("direction", "")
                    bleu4_score = char_bleu4(res_direction, sol_direction)
                    reward += bleu4_score
                    reward_branch_stat['scroll'] += 1
                else:
                    raise ValueError(f"Unresolved action type: {res_single['action_type']}")
            except Exception as e:
                print("Catch Exception while parsing the sampled action", e)
                reward = 0.0

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.0 for match in matches] # merged with acc


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # Load the dataset
    data_ori = json.load(open(script_args.dataset_positive_name, 'r'))
    data = [{'conversations': x, 'bad_step': False} for x in data_ori]
    # no neg test.
    data_failed = json.load(open(script_args.dataset_negative_name, 'r'))
    data += [{'conversations': x, 'bad_step': True} for x in data_failed]

    dataset = Dataset.from_list(data)
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
    print('='*15, "count reward branch", '='*15)
    print(reward_branch_stat) # cnt reward branch
    print('='*45)
