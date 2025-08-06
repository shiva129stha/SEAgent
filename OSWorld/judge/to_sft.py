import os
import json
import shutil
import glob
import copy
cnt = 0

for software in ['vs_code', 'libreoffice_impress', 'libreoffice_writer']:
    res_chat = []
    for idx in os.listdir(f'results_en_phase0_multi_init/pred_corr_{software}'):
        for index in os.listdir(f'results_en_phase0_multi_init/pred_corr_{software}/{idx}'):
            example_path = f'results_en_phase0_multi_init/pred_corr_{software}/{idx}/{index}'
            try:
                chat_ori = json.load(open(os.path.join(example_path, "messages.txt"), "r"))
                chat_answer_ori = json.load(open(os.path.join(example_path, "process.txt"), "r"))
                chat = copy.deepcopy(chat_ori)
                chat_answer = copy.deepcopy(chat_answer_ori)
                for conv, answer in zip(chat, chat_answer):
                    conv.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]})
                    res_chat.append(conv)
            except:
                continue
    print(len(res_chat))
    json.dump(res_chat, open(os.path.join(f"{software}_sft.json"), "w"), indent=4)
