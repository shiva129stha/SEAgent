import base64
import json
from pathlib import Path

from . import defaults, utils, existing


def format_content_for_image(b64_url: str, text: str) -> list:
    """
    Wrap a base-64 image string and a short caption in the structure expected by the
    chat API.

    Parameters
    ----------
    b64_url : str
        Base-64 data-URL for the image.
    text : str
        Caption that will appear above the image (e.g. “Here is the initial screenshot.”).

    Returns
    -------
    list
        A list with one text block and one image block, ready to be appended to the
        `messages` payload.
    """
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": b64_url}},
    ]

def format_steps(steps: list, step_template: str = None) -> str:
    if step_template is None:
        step_template = defaults.STEP_TEMPLATE

    steps_str = ""
    for i, step in enumerate(steps):
        steps_str += step_template.format(
            step_number=i + 1,
            url=step["url"],
            action=step["action"],
            reasoning=step["reasoning"],
        )
    return steps_str


def format_chat_messages_for_judge(
    sys_prompt: str,
    goal_msg: str,
    action_msg: str,
    axtree_msg: str,
    img_msg_content: list,
    final_msg: str = None,
):
    if axtree_msg is None:
        axtree_msg_content = []
    else:
        axtree_msg_content = [{"type": "text", "text": axtree_msg}]

    if final_msg is None:
        final_msg = defaults.FINAL_MSG

    user_content_lst = [
        {"type": "text", "text": goal_msg},
        {"type": "text", "text": action_msg},
        *axtree_msg_content,
        *img_msg_content,
        {"type": "text", "text": final_msg},
    ]

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content_lst},
    ]


def get_response_msg(response: dict):
    return response["choices"][0]["message"]["content"]


def get_content_inside_tag(tag: str, response_msg: str):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"

    start_idx = response_msg.find(start_tag)
    end_idx = response_msg.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return None
    return response_msg[start_idx + len(start_tag) : end_idx]


def parse_judgment(response_msg: dict):
    # get the content between <reasoning>, <success>, <side>, <optimal>, <loop>
    reasoning = get_content_inside_tag("reasoning", response_msg)
    success = get_content_inside_tag("success", response_msg)
    side = get_content_inside_tag("side", response_msg)
    optimal = get_content_inside_tag("optimal", response_msg)
    loop = get_content_inside_tag("loop", response_msg)

    return {
        "reasoning": reasoning,
        "trajectory_success": success,
        "trajectory_side_effect": side,
        "trajectory_optimality": optimal,
        "trajectory_looping": loop,
    }


def parse_aer_judgment(response_msg: dict):
    judgment = {
        "reasoning": None,
        "trajectory_success": None,
        "trajectory_side_effect": "n/a",
        "trajectory_optimality": "n/a",
        "trajectory_looping": "n/a",
    }
    for line in response_msg.split("\n"):
        if line.startswith("Thoughts:"):
            splitted = line.split(":", 1)
            if len(splitted) == 2:
                judgment["reasoning"] = splitted[1].strip()
        elif line.startswith("Status:"):
            splitted = line.split(":", 1)
            if len(splitted) == 2:
                judgment["trajectory_success"] = splitted[1].strip()

    return judgment

def convert_likert5_to_likert4(score_on_5: str) -> str:
    # first, convert to integers
    try:
        score_on_5 = int(score_on_5.strip())
    except ValueError:
        raise ValueError(f"Invalid Likert-5 value: {score_on_5}")
    
    if score_on_5 == 1:
        return 1
    elif score_on_5 == 2:
        return 1
    elif score_on_5 == 3:
        return 2
    elif score_on_5 == 4:
        return 3
    elif score_on_5 == 5:
        return 4

def convert_likert5_to_binary(score_on_5: str) -> str:
    # first, convert to integers
    try:
        score_on_5 = int(score_on_5.strip())
    except ValueError:
        raise ValueError(f"Invalid Likert-5 value: {score_on_5}")
    
    if score_on_5 >= 4:
        return 1
    else:
        return 0
        
def parse_nnetnav_judgment(response_msg: dict):
    judgment = {
        "reasoning": None,
        "trajectory_success": None,
        "trajectory_side_effect": "n/a",
        "trajectory_optimality": "n/a",
        "trajectory_looping": "n/a",
    }
    if "Reward:" in response_msg:
        # split by "Success:"
        reasoning_line, success_line = response_msg.split("Reward:", 1)
        success_line = success_line.strip()
        reasoning_line = reasoning_line.strip()

        # if success_line is an integer, convert it to a boolean
        if success_line.isdigit():
            judgment["trajectory_optimality"] = convert_likert5_to_likert4(success_line)
            judgment["trajectory_success"] = convert_likert5_to_binary(success_line)
        else:
            breakpoint()
            raise ValueError(f"Invalid success line: {success_line}")
        
        if "Thought:" in reasoning_line:
            # split by "Thoughts:"
            _, reasoning_line = reasoning_line.split("Thought:", 1)
        reasoning_line = reasoning_line.strip()
        judgment["reasoning"] = reasoning_line
    
    if judgment["trajectory_success"] is None or judgment["reasoning"] is None:
        breakpoint()
    return judgment


def create_chat_messages_from_trajectory_ori(
    trajectory, traj_dir, frame_num=1, use_screenshot=True, use_axtree=True, invert_system_prompt=False
):
    last_step = trajectory["steps"][-1]

    if use_screenshot:
        img_msg_content = format_content_for_image(
            utils.image_to_base64(last_step["screenshot_path"])
        )
    else:
        img_msg_content = []

    if use_axtree:
        axtree_msg = defaults.AXTREE_TEMPLATE.format(axtree=last_step["axtree"])
        axtree_pruned = defaults.AXTREE_TEMPLATE.format(
            axtree=last_step["axtree_pruned"]
        )
    else:
        axtree_msg = None
        axtree_pruned = None
    
    if invert_system_prompt:
        sys_prompt = defaults.INVERTED_SYSTEM_PROMPT
    else:
        sys_prompt = defaults.SYSTEM_PROMPT
    
    action_msg = defaults.ACTION_TEMPLATE.format(
        steps=format_steps(trajectory["steps"])
    )
    
    chat_messages = format_chat_messages_for_judge(
        sys_prompt=sys_prompt,
        goal_msg=defaults.GOAL_TEMPLATE.format(goal=trajectory["goal"]),
        action_msg=action_msg,
        axtree_msg=axtree_msg,
        img_msg_content=img_msg_content,
    )

    chat_messages_pruned = format_chat_messages_for_judge(
        sys_prompt=sys_prompt,
        goal_msg=defaults.GOAL_TEMPLATE.format(goal=trajectory["goal"]),
        action_msg=action_msg,
        axtree_msg=axtree_pruned,
        img_msg_content=img_msg_content,
    )

    return {
        "regular": chat_messages,
        "pruned": chat_messages_pruned,
    }


def create_chat_messages_from_trajectory(
    trajectory, traj_dir, frame_num=1, use_screenshot=True, use_axtree=True, invert_system_prompt=False
):
    """
    Build the message list that will be sent to the judge LLM.

    The `frame_num` argument controls how many screenshots are attached:
    * 1 – last step only (status-quo behaviour)
    * 2 – first + last step
    * >2 – first, last, and `frame_num-2` evenly-spaced intermediate frames

    For every attached screenshot we prepend a short caption:
        • first frame  →  “Here is the initial screenshot.”
        • last frame   →  “Here is the screenshot of the last step.”
        • middle frame →  “Here is the step{k} screenshot.”  (k is the 1-based index
                           in the trajectory, *not* the sampling order)
    """
    steps = trajectory["steps"]
    n_steps = len(steps)

    # -------- Select which steps to sample --------
    if not use_screenshot or frame_num <= 0:
        sampled_indices = []
    elif frame_num == 1 or n_steps == 1:
        sampled_indices = [n_steps - 1]                       # last only
    else:
        # linspace-like selection that always includes first (0) and last (n_steps-1)
        frame_num = min(frame_num, n_steps)
        sampled_indices = [
            round(i * (n_steps - 1) / (frame_num - 1)) for i in range(frame_num)
        ]
        # Ensure uniqueness in rare rounding collisions
        sampled_indices = sorted(set(sampled_indices))

    # -------- Build image message content --------
    img_msg_content = []
    for idx in sampled_indices:
        step = steps[idx]
        if idx == 0:
            caption = "Here is the screenshot of the inital state."
        elif idx == n_steps - 1:
            caption = "Here is the screenshot of the final state."
        else:
            caption = f"Here is the screenshot agter {idx + 1} step."
        img_msg_content += format_content_for_image(
            utils.image_to_base64(step["screenshot_path"]), caption
        )

    # -------- a11y tree handling --------
    if use_axtree:
        axtree_full = defaults.AXTREE_TEMPLATE.format(axtree=steps[-1]["axtree"])
        axtree_pruned = defaults.AXTREE_TEMPLATE.format(
            axtree=steps[-1]["axtree_pruned"]
        )
    else:
        axtree_full = axtree_pruned = None

    # -------- system prompt selection --------
    sys_prompt = (
        defaults.INVERTED_SYSTEM_PROMPT if invert_system_prompt else defaults.SYSTEM_PROMPT
    )

    sys_prompt = sys_prompt if frame_num == 1 else (defaults.SYSTEM_PROMPT_PROCESS)

    # -------- action trace --------
    action_msg = defaults.ACTION_TEMPLATE.format(steps=format_steps(steps))

    # -------- assemble two versions (full + pruned) --------
    chat_messages = format_chat_messages_for_judge(
        sys_prompt=sys_prompt,
        goal_msg=defaults.GOAL_TEMPLATE.format(goal=trajectory["goal"]),
        action_msg=action_msg,
        axtree_msg=axtree_full,
        img_msg_content=img_msg_content,
    )

    chat_messages_pruned = format_chat_messages_for_judge(
        sys_prompt=sys_prompt,
        goal_msg=defaults.GOAL_TEMPLATE.format(goal=trajectory["goal"]),
        action_msg=action_msg,
        axtree_msg=axtree_pruned,
        img_msg_content=img_msg_content,
    )

    return {
        "regular": chat_messages,
        "pruned": chat_messages_pruned,
    }
