import os
import json
import re
import glob
import re
import ast

def escape_single_quotes(text):
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
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
        node = ast.parse(action_str, mode='eval')

        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        call = node.body

        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str): 
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
            def escape_quotes(match):
                content = match.group(1)
                return content
            pattern = r"type\(content='(.*?)'\)" 
            content = re.sub(pattern, escape_quotes, action_str)

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

        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                float_numbers = [float(num) / factor for num in numbers]
                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions

corr = 0
mid_coor = 0
all = 0
sft = []
error_sft = []
max_image = 1
model = "gui_judge"
mid_pth = os.environ['MID_PATH']
base_judge_dir = f"judge_res/judge_{model}/{mid_pth}"
base_result_dir = f"{mid_pth}/pyautogui/screenshot_a11y_tree/ui_tars_7b_dpo/vscode"

for judge_id in os.listdir(base_judge_dir):
    try:
        res = json.load(open(os.path.join(base_judge_dir, judge_id), "r"))
    except:
        continue

    judge_id_no_ext = judge_id.replace(".json", "")
    process_pth = os.path.join(base_result_dir, judge_id_no_ext)

    if res.get('Correctness') and res.get('Optimized'):
        corr += 1
        try:
            messages = json.load(open(os.path.join(process_pth, "messages.txt"), 'r'))
            responses = json.load(open(os.path.join(process_pth, "process.txt"), 'r'))
            assert len(messages) == len(responses)
        except:
            continue

        for i, response in enumerate(responses):
            try:
                parsed_response = parse_action_qwen2vl(response, factor=1000, image_height=1080, image_width=1920)
                pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_response,
                    image_height=1080, image_width=1920,
                    input_swap=False
                )
            except:
                print("skip failed pygui code")
                continue

            if response != []:
                num_images = 0
                for conv in messages[i]:
                    if conv['content'][0]['type'] == 'image_url':
                        num_images += 1
                for j in range(num_images - max_image):
                    for t in range(len(messages[i])):
                        if messages[i][t]['content'][0]['type'] == 'image_url':
                            del messages[i][t]
                            break
                print(f'del exceed history images: {num_images - max_image}')
                check_max = len(messages[i])
                check_i = 0
                while check_i < len(messages[i]):
                    if messages[i][check_i]['role'] == 'assistant':
                        try:
                            parsed_response = parse_action_qwen2vl(messages[i][check_i]['content'][0]['text'], factor=1000, image_height=1080, image_width=1920)
                            pyautogui_code = parsing_response_to_pyautogui_code(
                                parsed_response,
                                image_height=1080, image_width=1920,
                                input_swap=False
                            )
                            check_i += 1
                        except:
                            print("skip failed pygui code in history record!")
                            del messages[i][check_i]
                    else:
                        check_i += 1
                sft.append(
                    messages[i] + [{'role': 'assistant', 'content': [{'type': 'text', 'text': response}]}]
                )

    elif res.get('First_Error_Step') is not None and type(res['First_Error_Step']) == int and res['First_Error_Step'] > 2:
        if len(res.get('Redundant')) == 0 or (len(res.get('Redundant')) > 0 and res.get('First_Error_Step') < res.get('Redundant')[0]):
            error_step_index = int(res['First_Error_Step'])
        else:
            error_step_index = res.get('Redundant')[0]
        if error_step_index < 2:
            continue
        try:
            messages = json.load(open(os.path.join(process_pth, "messages.txt"), 'r'))
            responses = json.load(open(os.path.join(process_pth, "process.txt"), 'r'))
            assert len(messages) == len(responses)
        except:
            print(f"skip mid-coor due to loading error: {judge_id}")
            continue

        image_dir = process_pth + "/"
        image_files = glob.glob(os.path.join(image_dir, "step_*_*.png"))

        def extract_k(file_path):
            match = re.search(r"step_(\d+)_", os.path.basename(file_path))
            return int(match.group(1)) if match else float('inf')

        image_files_sorted = sorted(image_files, key=extract_k)

        if error_step_index < len(image_files_sorted):
            error_image_k = extract_k(image_files_sorted[error_step_index])
            print(error_image_k, image_files_sorted[error_step_index])
            print(f"[{judge_id}] First Error Step = {error_step_index}, Image k = {error_image_k}")
        else:
            print(f"[{judge_id}] Error step index {error_step_index} out of range ({len(image_files_sorted)})")
            continue
        mid_coor += 1

        for i in range(error_step_index - 1):
            response = responses[i]
            try:
                parsed_response = parse_action_qwen2vl(response, factor=1000, image_height=1080, image_width=1920)
                pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_response,
                    image_height=1080, image_width=1920,
                    input_swap=False
                )
            except:
                print("skip failed pygui code")
                continue
            if response == []:
                continue
            num_images = 0
            for conv in messages[i]:
                if conv['content'][0]['type'] == 'image_url':
                    num_images += 1
            for j in range(num_images - max_image):
                for t in range(len(messages[i])):
                    if messages[i][t]['content'][0]['type'] == 'image_url':
                        del messages[i][t]
                        break
            check_max = len(messages[i])
            check_i = 0
            while check_i < len(messages[i]):
                if messages[i][check_i]['role'] == 'assistant':
                    try:
                        parsed_response = parse_action_qwen2vl(messages[i][check_i]['content'][0]['text'], factor=1000, image_height=1080, image_width=1920)
                        pyautogui_code = parsing_response_to_pyautogui_code(
                            parsed_response,
                            image_height=1080, image_width=1920,
                            input_swap=False
                        )
                        check_i += 1
                    except:
                        print("skip failed pygui code in history record!")
                        del messages[i][check_i]
                else:
                    check_i += 1
            sft.append(
                messages[i] + [{'role': 'assistant', 'content': [{'type': 'text', 'text': response}]}]
            )

            i = error_step_index - 1
            response = responses[i]
            try:
                parsed_response = parse_action_qwen2vl(response, factor=1000, image_height=1080, image_width=1920)
                pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_response,
                    image_height=1080, image_width=1920,
                    input_swap=False
                )
            except:
                print("skip failed pygui code")
                continue
            if response == []:
                continue
            num_images = 0
            for conv in messages[i]:
                if conv['content'][0]['type'] == 'image_url':
                    num_images += 1
            for j in range(num_images - max_image):
                for t in range(len(messages[i])):
                    if messages[i][t]['content'][0]['type'] == 'image_url':
                        del messages[i][t]
                        break
            check_max = len(messages[i])
            check_i = 0
            while check_i < len(messages[i]):
                if messages[i][check_i]['role'] == 'assistant':
                    try:
                        parsed_response = parse_action_qwen2vl(messages[i][check_i]['content'][0]['text'], factor=1000, image_height=1080, image_width=1920)
                        pyautogui_code = parsing_response_to_pyautogui_code(
                            parsed_response,
                            image_height=1080, image_width=1920,
                            input_swap=False
                        )
                        check_i += 1
                    except:
                        print("skip failed pygui code in history record!")
                        del messages[i][check_i]
                else:
                    check_i += 1
            error_sft.append(
                messages[i] + [{'role': 'assistant', 'content': [{'type': 'text', 'text': response}]}]
            )
    all += 1

print(len(sft))
json.dump(sft, open(f'7b_positive_traj_w_mid_maximage={max_image}.json', 'w'), indent=4)
print(corr, mid_coor, all)

print(len(error_sft))
json.dump(error_sft, open(f'7b_negative_traj_w_mid_maximage={max_image}.json', 'w'), indent=4)
