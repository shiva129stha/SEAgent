import os
import re
import json
from openai import OpenAI
import base64
import ast
import argparse
from tqdm import tqdm
# task buffer update strategy
'''
1. Ask actor to generate captions for the screen shot (with corresponding action).
2. provide previous success&failure list.
3. Ask 4o to generate new tasks based on the screen shot and assess the previous success&failure list.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=int, default=-1, help='Current training phase')
parser.add_argument('--software', type=str, default='vscode', help='Software name')
parser.add_argument('--judge_model', type=str, default='gpt-4o-2024-11-20', help='Judge model name')
parser.add_argument('--task_generator', type=str, default='qwen72b', help='Judge model name')
parser.add_argument('--base_result_dir', type=str, default='results_en_phase_qwen', help='Base path to result dir')

args = parser.parse_args()

phase = args.phase
software = args.software
judge_model = args.judge_model
last_iter_pth = f"{args.base_result_dir}{phase}/7b_new_traj_multi_env=8_{software}_new_fix_nimg=1_maxtraj=15_t=0.0_r0/pyautogui/screenshot_a11y_tree/ui_tars_7b_dpo/exam_new"

first_round = (phase == -1)
current_task_buffer = None

if not first_round:
    current_task_buffer = json.load(open(f'task_buffer/task_buffer_uitars_7b_{software}_phase{phase}.json', 'r'))

client_judge_qwen = OpenAI(
    base_url="http://your.WorldStateModel.ip.address:port/v1",
    api_key="empty"
)

client_task_generator = OpenAI(
    base_url="http://your.QWEN_LLM.ip.address:port/v1/",
    api_key="empty"
)

image_data = []

exam = dict()
action_decription_list = []
document = ""
def build_prompt(action_decription_list, document, exam, prev_states):
    prompt = f"""
    You are now a teacher training a Computer Use Agent (CUA). This CUA is exposed to a new software environment and undergoes multiple rounds of iterative training. Your task is to issue new tasks for the agent to explore and train on, based on the feedback from the agent's actions. You are also responsible for summarizing a software usage manual to help the agent remember knowledge about the software.

    The agent has provided the following feedback on its operations within the software: {json.dumps(action_decription_list)}

    Here is the software usage document you summarized in the previous round: {document}

    And here is the agent's performance on the task you provided in the previous round: {json.dumps(exam)}

    Your are also access to the previous given tasks with the screenshot caption after agent's execution. You can also use these captions and results to evaluate the agent's capability and generate new task and update document accordingly given the caption of the new screen and the corresponding intruction with judged evaluation: {json.dumps(prev_states)}

    Please:
    - Analyze the agent's performance.
    - Integrate new knowledge from the feedback.
    - Update the usage manual accordingly.
    - Design a new set of tasks (with increased difficulty) (30 or more) that reinforce the concepts the agent struggled with in the last round.
    - Each task **must be concise and specific**, targeting a concrete atomic action, based on the document and agent's observations, such as:
        - "Create a file named `main.py`."
        - "Open `Terminal` card." 
    - Each task must be executable from software initial state with no file open, e.g. you should not generate task like `save xxx.txt` if xxx.txt doesn't exist or created.
    - if task is in sequencial order with reliance, you should output a seq list like [subtask1, subtask2, ...], if there is no reliance, output [task].
    - Decompose and target previous errors in a more focused way.

    Output your reasoning and analysis process first. Then output the updated usage document and task list in the following JSON format within a SINGLE JSON DICT easier for me to parse:
    ```json
    {{
        "software_document_new": "...",
        "exam_new": [[subtask1, subtask2, ...], [task]...]
    }}
    ```
    """
    return prompt

if first_round:
    with open(f"task_buffer/{software}_init.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        image_data.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}})
    prompt_text = f"what can you see in this opened software {software}, describe it in English as detail as possible. Describe all the elements, buttons you have seen."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}] + image_data}]
    response = client_judge_qwen.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=1
    )
    action_decription_list = {"open the software": response.choices[0].message.content}
    prompt_text = build_prompt(action_decription_list=action_decription_list, document="EMPTY FOR YOU TO BUILD", exam=dict(), prev_states=dict())
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    while True:
        try:
            response = client_task_generator.chat.completions.create(
                model=args.task_generator,
                messages=messages,
                temperature=1
            )
            main_text = response.choices[0].message.content
            main_text_json = main_text.replace('```json\n', 'SPLIT').replace('```json', 'SPLIT').split('SPLIT')[1]
            res = ast.literal_eval(main_text_json.replace('json\n', '').replace('`', ''))
            new_res = dict()
            new_res['software_document'] = res['software_document_new']
            new_res['exam'] = res['exam_new']
            break
        except Exception as e:
            print(f"Error: {e}")
            print(main_text_json)
    response = client_task_generator.chat.completions.create(
        model=args.task_generator,
        messages=messages,
        temperature=1
    )
else:
    prev_states = list()
    for index, episode in tqdm(enumerate(current_task_buffer['exam'])):
        state_pth = f"{last_iter_pth}/{index}"
        for i, task in enumerate(episode):
            if os.path.exists(f'{last_iter_pth}/{index}/step_{i}_final.png'):
                with open(f"task_buffer/{software}_init.png", "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    image_data = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}]        
                with open(f'{last_iter_pth}/{index}/step_{i}_final.png', "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    image_data += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}]
                prompt_text = f"These two images are the initial states and the final states after executing {task} from the agent. What can you see as something new appear compare to original states? Describe it in English as detail as possible. Describe all the elements, buttons you have seen that makes differences. Judge its correctness of whether the agent successfully conduct instruction: {task} with format <judge>success/fail</judge> and put your detailed reasoning process in format <think> evidence for judgement. </think>."
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}] + image_data}]
                response = client_judge_qwen.chat.completions.create(
                    model="/fs-computility/mllm/shared/mllm_ckpts/models--Qwen--Qwen2.5-VL-7B-Instruct",
                    messages=messages,
                    temperature=1
                )
                prev_states.append({
                    'instruction': task,
                    'screen_caption': response.choices[0].message.content,
                    'task_success': True if "<judge>success</judge>" in str(response.choices[0].message.content).lower() else False
                })
    while True:
        try:
            prompt_text = build_prompt(action_decription_list=action_decription_list, document=current_task_buffer['software_document'], exam=dict(), prev_states=prev_states)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            response = client_task_generator.chat.completions.create(
                model=args.task_generator,
                messages=messages,
                temperature=1
            )
            main_text = response.choices[0].message.content
            main_text_json = main_text.replace('```json\n', 'SPLIT').replace('```json', 'SPLIT').split('SPLIT')[1]
            res = ast.literal_eval(main_text_json.replace('json\n', '').replace('`', ''))
            new_res = dict()
            new_res['software_document'] = res['software_document_new']
            new_res['exam'] = res['exam_new']
            break
        except Exception as e:
            print(f"Error: {e}")
            print(main_text_json)

json.dump(new_res, open(f'task_buffer/task_buffer_qwen_{software}_phase{phase+1}.json', 'w'), indent=4)
