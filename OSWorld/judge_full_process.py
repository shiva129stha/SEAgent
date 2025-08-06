import os
import glob
import re
import json
import base64
import time
from openai import OpenAI
import ast
from PIL import Image
from io import BytesIO

model = "gui_judge"

if model == "4o":
    client = OpenAI(
        base_url="Your API url",
        api_key="Your API key",
    )
else:
    client = OpenAI(
        base_url="vllm deployed World State Model",
        api_key="empty",
    )
mid_pth = os.environ['MID_PATH']
phase_data = json.load(open("task_buffer/task_buffer_qwen_vscode_phase0.json", "r"))

for index, instruction in enumerate(phase_data['exam']):
    instruction = instruction[0] # use the first instruction for now.
    os.makedirs(f"judge_res/judge_{model}/{mid_pth}", exist_ok=True)
    if os.path.exists(f"judge_res/judge_{model}/{mid_pth}/{index}.json"):
        continue
    image_dir = f"{mid_pth}/pyautogui/screenshot_a11y_tree/ui_tars_7b_dpo/vscode/{index}"
    image_files = glob.glob(os.path.join(image_dir, "step_*_*.png"))

    def extract_k(file_path):
        match = re.search(r"step_(\d+)_", os.path.basename(file_path))
        return int(match.group(1)) if match else float('inf')

    image_files.sort(key=extract_k)

    prompt_text = (
                "I am evaluating the performance of a UI agent. The images provided are **sequential keyframes** that represent "
                "the full execution trajectory of the agent when attempting to follow a command. "
                f"These keyframes correspond to the instruction: **'{instruction}'**.\n\n"

                "Please thoroughly analyze the sequence to assess the following aspects:\n"
                "1. **Correctness** — Did the agent successfully complete the task as instructed?\n"
                "2. **Redundant Steps** — Identify any unnecessary or repeated actions that do not contribute to the goal.\n"
                "3. **Optimization** — Did the agent follow an efficient plan with a minimal number of steps?\n"
                "4. **First Error Step** — If the execution is incorrect or sub-optimal, determine the index of the **first keyframe where a mistake occurred**.\n"
                "5. **Error Analysis** — Provide a brief explanation of the mistake at that step.\n"
                "6. **Correct Action Suggestion** — Explain what the agent **should have done instead** at the point of error.\n\n"

                "**Important Instructions:**\n"
                "- The agent may have made progress toward the goal, but unless the task is **fully and correctly completed**, you must set 'Correctness' to **False**.\n"
                "- Be cautious in determining success. Missing confirmation screens, skipped inputs, or wrong UI elements clicked all count as errors.\n"
                "- Carefully examine all UI changes, button interactions, text entries, and any visual feedback in the screenshots.\n"
                "- Clearly indicate **which exact steps are redundant** (starting from 1).\n\n"

                "Once you finish the analysis, return your evaluation in the following dictionary format (include your step-by-step reasoning **above** the result):\n\n"
                "<analysis process>\n"
                "<res_dict>{\n"
                "  \"Correctness\": True/False,\n"
                "  \"Redundant\": [step_num, ...],\n"
                "  \"Optimized\": True/False,\n"
                "  \"First_Error_Step\": step_num or None,\n"
                "  \"Error_Type\": \"brief description of the mistake\",\n"
                "  \"Correct_Action\": \"what should have been done instead\"\n"
                "}</res_dict>"
            )

    image_data = []
    for file_path in image_files:
        try:
            with open(file_path, "rb") as image_file:
                image = Image.open(image_file)

                new_size = (int(image.width / 1.5), int(image.height / 1.5))
                image = image.resize(new_size)

                buffer = BytesIO()
                image.save(buffer, format="PNG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_data.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}})
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except Exception as e:
            print(f"⚠️ Error reading the file: {e}")

    if not image_data:
        print("❌ Empty Image in folder!")
        continue

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}] + image_data}]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1
        )

        print(f"\n🚀 {model} Keyframe Analysis Result:")
        print(response.choices[0].message.content)
        ori = response.choices[0].message.content
    except:
        continue
    try:
        if '<res_dict>' in ori:
            res_dict = ast.literal_eval(ori.split('<res_dict>')[1].replace("</res_dict>", ""))
        elif '```json' in ori:
            res_dict = ast.literal_eval(ori.split('```json')[1].replace("```", ""))
        elif '```python' in ori:
            res_dict = ast.literal_eval(ori.split('```python')[1].replace("```", ""))
        else:
            raise ValueError
    except:
        continue
    res_dict['ori'] = ori
    json.dump(res_dict, open(f"judge_res/judge_{model}/{mid_pth}/{index}.json", 'w'), indent=4)
    time.sleep(1)
