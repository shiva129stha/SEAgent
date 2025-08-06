import os
import glob
import re
import json
import base64
import time
import ast
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from itertools import cycle

# 多个 base_url 支持负载均衡
BASE_URLS = [
    "http://101.126.156.90:15360/v1",
    "http://101.126.156.90:64399/v1",
    "http://101.126.156.90:5199/v1",
    "http://101.126.156.90:49094/v1",
    "http://101.126.156.90:13470/v1",
    "http://101.126.156.90:44098/v1"
]
client_cycle = cycle([OpenAI(base_url=url, api_key="EMPTY") for url in BASE_URLS])

scale = {"1.0": (1920, 1080), "1.2": (1600, 900), "1.5": (1280, 720), "2.0": (960, 540)}
software_list = ['vs_code', 'libreoffice_impress', 'libreoffice_writer', 'vlc', 'gimp']
MAX_WORKERS = 24

def process_task(software, r, init_pth, i, instruction):
    init_id = init_pth.split('_')[0]

    # 第一轮结果文件
    base_res_path = f"results_en_phase0_multi_init/judge_res_{software}_r{r}/{init_id}/{i}.json"
    if not os.path.exists(base_res_path):
        return

    # 如果已经有投票结果就跳过
    result_path = f"results_en_phase0_multi_init/voted4_judge_res_{software}_r{r}/{init_id}/{i}.json"
    if os.path.exists(result_path):
        return

    # 读取第一轮结果
    try:
        first_res = json.load(open(base_res_path))
    except:
        return

    if not first_res.get("Correctness", False):
        return  # 如果第一轮就错了，直接跳过，无需投票

    # 构造图片路径
    image_dir = f"results_en_phase0_multi_init/7b_new_traj_{software}_maxtraj=15_r{r}/pyautogui/screenshot_a11y_tree/ui-tars/{software}/{init_id}/{i}"
    if not os.path.exists(image_dir):
        return

    all_image_files = glob.glob(os.path.join(image_dir, "step_*_*.png"))
    step_image_dict = {}
    for file_path in all_image_files:
        match = re.search(r"step_(\d+)_", os.path.basename(file_path))
        if match:
            step_k = int(match.group(1))
            if step_k not in step_image_dict:
                step_image_dict[step_k] = file_path

    image_files = [step_image_dict[k] for k in sorted(step_image_dict.keys())]
    if not image_files:
        return

    # 构造 prompt
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
        "You output must be extremly concise and focused, with clear emphasis on key points. If the agent fails, only provide the core reason for the first step failure, ignoring other minor issues. Keep the language clear and direct."
        "Once you finish the analysis, return your evaluation in the following dictionary JSON format:\n\n"
        "<captions>\n"
        "Frame1: caption of the first frame\n"
        "Frame2: caption of the second frame\n"
        "(max 15 frame captions)\n"
        "</captions>"
        "<res_dict>{\n"
        "  \"Correctness\": True/False,\n"
        "  \"Redundant\": [step_num, ...],\n"
        "  \"Optimized\": True/False,\n"
        "  \"First_Error_Step\": step_num or None,\n"
        "  \"Error_Type\": \"brief description of the mistake\",\n"
        "  \"Correct_Action\": \"what should have been done instead\"\n"
        "}</res_dict>"
    )

    # 构造图片输入
    image_data = []
    for file_path in image_files:
        try:
            with Image.open(file_path) as img:
                resized_img = img.resize(scale["1.0"])
                buffered = io.BytesIO()
                resized_img.save(buffered, format="PNG")
                encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_data.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        except:
            return

    if not image_data:
        return

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}] + image_data}]
    all_results = [first_res]
    all_correct = True

    for attempt in range(3):  # 再跑3次
        client = next(client_cycle)
        try:
            response = client.chat.completions.create(
                model="qwen72b",
                messages=messages,
                temperature=1.0,
                top_p=0.6
            )
            ori = response.choices[0].message.content

            if '<res_dict>' in ori:
                res_dict = ast.literal_eval(ori.split('<res_dict>')[1].replace("</res_dict>", ""))
            elif '```json' in ori:
                res_dict = ast.literal_eval(ori.split('```json')[1].replace("```", ""))
            elif '```python' in ori:
                res_dict = ast.literal_eval(ori.split('```python')[1].replace("```", ""))
            else:
                all_correct = False
                continue

            res_dict['ori'] = ori
            all_results.append(res_dict)

            if not res_dict.get("Correctness", False):
                all_correct = False

            time.sleep(0.5)
        except Exception as e:
            print(f"❌ 投票失败：{software} | {init_id}-{i} | 第{attempt+2}轮错误：{e}")
            all_correct = False

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if all_correct and len(all_results) == 4:
        json.dump({
            "final_decision": True,
            "votes": all_results
        }, open(result_path, 'w'), indent=4)
    else:
        fail_path = result_path.replace("voted_judge_res", "voted_judge_fail")
        json.dump({
            "final_decision": False,
            "votes": all_results
        }, open(fail_path, 'w'), indent=4)


# 主入口：多 software + 多轮次 r
for software in software_list:
    task_base = f'task_buffer/task_buffer_uitars_7b_qwen2.5_winstr_hi/{software}'
    for r in range(5):
        tasks = []
        for init_pth in os.listdir(task_base):
            full_list = json.load(open(os.path.join(task_base, init_pth)))
            for i, instruction in enumerate(full_list['exam']):
                tasks.append((software, r, init_pth, i, instruction))

        print(f"🔧 Software={software}, Round={r}: Total {len(tasks)} tasks")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_task, *t) for t in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{software}-r{r}"):
                pass
