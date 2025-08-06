import ast
import base64
import logging
import math
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List

import backoff
import numpy as np
from PIL import Image
from requests.exceptions import SSLError
import openai
from openai import OpenAI
from google.api_core.exceptions import (
    BadRequest,
    InternalServerError,
    InvalidArgument,
    ResourceExhausted,
)

from mm_agents.accessibility_tree_wrap.heuristic_retrieve import (
    filter_nodes,
)
from mm_agents.prompts import (
    UITARS_ACTION_SPACE,
    UITARS_CALL_USR_ACTION_SPACE,
    UITARS_USR_PROMPT_NOTHOUGHT,
    UITARS_USR_PROMPT_THOUGHT,
)

from mm_agents.utils import parsing_response_to_pyautogui_code, parse_action_to_structure_output, smart_resize

logger = logging.getLogger("desktopenv.agent")

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

pure_text_settings = ["a11y_tree"]

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py

# 定义一个函数来解析每个 action
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
        print(f"Failed to parse action '{action_str}': {e}")
        return None
    
def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

# def parse_action_qwen2vl(text, factor, image_height, image_width):
#     text = text.strip()
#     # 正则表达式匹配 Action 字符串
#     if text.startswith("Thought:"):
#         thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
#         thought_hint = "Thought: "
#     elif text.startswith("Reflection:"):
#         thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
#         thought_hint = "Reflection: "
#     elif text.startswith("Action_Summary:"):
#         thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
#         thought_hint = "Action_Summary: "
#     else:
#         thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
#         thought_hint = "Thought: "
#     reflection, thought = None, None
#     thought_match = re.search(thought_pattern, text, re.DOTALL)
#     if thought_match:
#         if len(thought_match.groups()) == 1:
#             thought = thought_match.group(1).strip()
#         elif len(thought_match.groups()) == 2:
#             thought = thought_match.group(2).strip()
#             reflection = thought_match.group(1).strip()
#     assert "Action:" in text
#     action_str = text.split("Action:")[-1]

#     tmp_all_action = action_str.split("\n\n")
#     all_action = []
#     for action_str in tmp_all_action:
#         if "type(content" in action_str:
#             # 正则表达式匹配 content 中的字符串并转义单引号
#             def escape_quotes(match):
#                 content = match.group(1)  # 获取 content 的值
#                 return content

#             # 使用正则表达式进行替换
#             pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
#             content = re.sub(pattern, escape_quotes, action_str)

#             # 处理字符串
#             action_str = escape_single_quotes(content)
#             action_str = "type(content='" + action_str + "')"
#         all_action.append(action_str)

#     parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
#     actions = []
#     for action_instance, raw_str in zip(parsed_actions, all_action):
#         if action_instance == None:
#             print(f"Action can't parse: {raw_str}")
#             continue
#         action_type = action_instance["function"]
#         params = action_instance["args"]

#         # import pdb; pdb.set_trace()
#         action_inputs = {}
#         for param_name, param in params.items():
#             if param == "": continue
#             param = param.lstrip()  # 去掉引号和多余的空格
#             # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
#             action_inputs[param_name.strip()] = param
            
#             if "start_box" in param_name or "end_box" in param_name:
#                 ori_box = param
#                 # Remove parentheses and split the string by commas
#                 numbers = ori_box.replace("(", "").replace(")", "").split(",")

#                 # Convert to float and scale by 1000
#                 float_numbers = [float(num) / factor for num in numbers]
#                 if len(float_numbers) == 2:
#                     float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
#                 action_inputs[param_name.strip()] = str(float_numbers)

#         # import pdb; pdb.set_trace()
#         actions.append({
#             "reflection": reflection,
#             "thought": thought,
#             "action_type": action_type,
#             "action_inputs": action_inputs,
#             "text": text
#         })
#     return actions

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    ]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text
                if '"' not in node.text
                else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith(
            "EditWrapper"
        ) and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (
                node_text
                if '"' not in node_text
                else '"{:}"'.format(node_text.replace('"', '""'))
            )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                (
                    node.get("{{{:}}}class".format(_attributes_ns), "")
                    if platform == "ubuntu"
                    else node.get("{{{:}}}class".format(class_ns_windows), "")
                ),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get("{{{:}}}screencoord".format(_component_ns), ""),
                node.get("{{{:}}}size".format(_component_ns), ""),
            )
        )

    return "\n".join(linearized_accessibility_tree)

def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    # enc = tiktoken.encoding_for_model("gpt-4")
    # tokens = enc.encode(linearized_accessibility_tree)
    # if len(tokens) > max_tokens:
    #     linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
    #     linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree

class UITARSAgent:
    def __init__(
        self,
        model="ui_tars",
        platform="ubuntu",
        max_tokens=1000,
        top_p=0.9,
        top_k=1.0,
        temperature=0.0,
        action_space="pyautogui",
        observation_type="screenshot_a11y_tree",
        # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
        max_trajectory_length=50,
        a11y_tree_max_tokens=10000,
        history_n=5,
        runtime_conf: dict = {
            "infer_mode": "qwen2vl_user",
            "prompt_style": "qwen2vl_user",
            "input_swap": False, # false for current vm script
            "language": "English",
            "max_steps": 50,
            # "history_n": 5, # number of history images
            "screen_height": 1080,
            "screen_width": 1920
        }
    ):
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.runtime_conf = runtime_conf
        # change into UI_TARs.
        if model == "ui_tars_7b_dpo":
            self.vlm = OpenAI(
                base_url="http://101.126.156.90:7705/v1",
                api_key="empty",
            ) # should replace with your UI-TARS server api
        elif model == "ui_tars_1.5_7b": # still use 7005->8001 port
            self.vlm = OpenAI(
                base_url="http://101.126.156.90:12933/v1",
                # base_url="http://101.126.156.90:7705/v1",
                api_key="empty",
            ) # should replace with your UI-TARS server api
        else:
            assert model == "ui_tars_72b_dpo"
            self.vlm = OpenAI(
                base_url="http://101.126.156.90:47236/v1",
                api_key="empty",
            ) # should replace with your UI-TARS server api
        self.infer_mode = self.runtime_conf["infer_mode"]
        self.prompt_style = self.runtime_conf["prompt_style"]
        self.input_swap = self.runtime_conf["input_swap"]
        self.language = self.runtime_conf["language"]
        self.max_steps = self.runtime_conf["max_steps"]

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        
        self.prompt_action_space = UITARS_ACTION_SPACE
        self.customize_action_parser = parse_action_to_structure_output
        self.action_parse_res_factor = 1000
        if self.infer_mode == "qwen2vl_user":
            self.prompt_action_space = UITARS_CALL_USR_ACTION_SPACE
    
        self.prompt_template = UITARS_USR_PROMPT_THOUGHT
        
        if self.prompt_style == "qwen2vl_user":
            self.prompt_template = UITARS_USR_PROMPT_THOUGHT

        elif self.prompt_style == "qwen2vl_no_thought":
            self.prompt_template = UITARS_USR_PROMPT_NOTHOUGHT

        self.history_n = history_n

    def predict(
        self, instruction: str, obs: Dict, last_action_after_obs: Dict = None
    ) -> List:
        """
        Predict the next action(s) based on the current observation.
        """

        # Append trajectory
        # print(len(self.observations), len(self.actions), len(self.actions))
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(
            self.thoughts
        ), "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length :]
                _actions = self.actions[-self.max_trajectory_length :]
                _thoughts = self.thoughts[-self.max_trajectory_length :]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(
            _observations, _actions, _thoughts
        ):
            # {{{1
            if self.observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

            else:
                raise ValueError(
                    "Invalid observation_type type: " + self.observation_type
                )  # 1}}}

        if last_action_after_obs is not None and self.infer_mode == "double_image":
            self.history_images.append(last_action_after_obs["screenshot"])

        self.history_images.append(obs["screenshot"])

        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = obs["screenshot"]
            # try:
            #     linearized_accessibility_tree = (
            #         linearize_accessibility_tree(
            #             accessibility_tree=obs["accessibility_tree"],
            #             platform=self.platform,
            #         )
            #         if self.observation_type == "screenshot_a11y_tree"
            #         else None
            #     )
            # except:
            linearized_accessibility_tree = None
            # logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append(
                    {
                        "screenshot": base64_image,
                        "accessibility_tree": None, # not use tree.
                    }
                )
            else:
                self.observations.append(
                    {"screenshot": base64_image, "accessibility_tree": None}
                )

        else:
            raise ValueError(
                "Invalid observation_type type: " + self.observation_type
            )  # 1}}}

        if self.infer_mode == "qwen2vl_user":
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                action_space=self.prompt_action_space,
                language=self.language
            )
        elif self.infer_mode == "qwen2vl_no_thought":
            user_prompt = self.prompt_template.format(
                instruction=instruction
            )

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        max_pixels = 1350 * 28 * 28
        min_pixels = 100 * 28 * 28
        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")
        max_image_nums_under_32k = int(32768*0.75/max_pixels*28*28) # 32k image tokens is too long for training.
        if len(self.history_images) > max_image_nums_under_32k:
            num_of_images = min(5, len(self.history_images))
            max_pixels = int(32768*0.75) // num_of_images

        for turn, image in enumerate(self.history_images):
            if len(images) >= 5:
                break
            try:
                image = Image.open(BytesIO(image))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            height, width = smart_resize(image.height, image.width)
            image = image.resize((width, height))
            # if image.width * image.height > max_pixels:
            #     """
            #     如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
            #     """
            #     resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            #     width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            #     image = image.resize((width, height))
            # if image.width * image.height < min_pixels:
            #     resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            #     width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
            #     image = image.resize((width, height))

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):

                    cur_image = images[image_num]
                    encoded_string = pil_to_base64(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                    })
                    image_num += 1
                    
                messages.append({
                    "role": "assistant",
                    "content": [history_response]
                })

            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1
        
        else:
            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1

        try_times = 6
        while True:
            if try_times <= 0:
                print(f"Reach max retry times to fetch response from client, as error flag.")
                return "client error", ["DONE"], [], messages
            try:
                response = self.vlm.chat.completions.create(
                    model="ui-tars",
                    messages=messages,
                    # frequency_penalty=1,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    # extra_body={
                    #             "do_sample": True,
                    #             "top_p": self.top_p
                    #             }
                )
                prediction = response.choices[0].message.content.strip()
                parsed_responses = self.customize_action_parser(
                    prediction,
                    self.action_parse_res_factor,
                    self.runtime_conf["screen_height"],
                    self.runtime_conf["screen_width"]
                )
                break

            except Exception as e:
                print(f"Error when fetching response from client, with response: {response}")
                prediction = None
                try_times -= 1

        if prediction is None:
            return "client error", ["DONE"], [], messages

        self.history_responses.append({'type': 'text', 'text': re.sub(r"click\(start_box='(\([^\)]+\))'\)", r"click(start_box='<|box_start|>\1<|box_end|>')", prediction)}) # modify to correct type as dict.
        self.thoughts.append(prediction)

        try:
            parsed_responses = self.customize_action_parser(
                prediction,
                self.action_parse_res_factor,
                self.runtime_conf["screen_height"],
                self.runtime_conf["screen_width"]
            )
        except Exception as e:
            print(f"Parsing action error: {prediction}, with error:\n{e}")
            return f"Parsing action error: {prediction}, with error:\n{e}", ["DONE"], []

        actions = []
        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:

                if parsed_response["action_type"] == FINISH_WORD:
                    self.actions.append(actions)

                    return prediction, ["DONE"], [], messages
                
                elif parsed_response["action_type"] == WAIT_WORD:
                    self.actions.append(actions)
                    return prediction, ["WAIT"], [], messages
                
                elif parsed_response["action_type"] == ENV_FAIL_WORD:
                    self.actions.append(actions)
                    return prediction, ["FAIL"], [], messages

                elif parsed_response["action_type"] == CALL_USER:
                    self.actions.append(actions)
                    return prediction, ["FAIL"], [], messages

            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_response,
                self.runtime_conf["screen_height"],
                self.runtime_conf["screen_width"],
                self.input_swap
            )
            actions.append(pyautogui_code)

        self.actions.append(actions)

        if len(self.history_responses) >= self.max_trajectory_length:
            # Default to FAIL if exceed max steps
            actions = ["FAIL"]

        return prediction, actions, parsed_responses, messages


    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
            # General exceptions
            SSLError,
            # OpenAI exceptions
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            # Google exceptions
            InvalidArgument,
            ResourceExhausted,
            InternalServerError,
            BadRequest,
            # Groq exceptions
            # todo: check
        ),
        interval=30,
        max_tries=10,
    )
    
    def reset(self, runtime_logger):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
