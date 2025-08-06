import os
import json
import shutil
import glob

cnt = 0

for software in ['vs_code', 'libreoffice_impress', 'libreoffice_writer', 'vlc', 'gimp']:
    for r in range(5):    
        for init_pth in os.listdir(f'task_buffer/task_buffer_uitars_7b_qwen2.5_winstr_hi/{software}'):
            full_list = json.load(open(f'task_buffer/task_buffer_uitars_7b_qwen2.5_winstr_hi/{software}/{init_pth}', 'r'))
            init_id = init_pth.split('_')[0]
            for i, instruction in enumerate(full_list['exam']):
                if os.path.exists(f"results_en_phase0_multi_init/judge_res_{software}_r{r}/{init_id}/{i}.json"):
                    judge_res = json.load(open(f"results_en_phase0_multi_init/judge_res_{software}_r{r}/{init_id}/{i}.json", 'r'))
                    if judge_res['Correctness'] == True and not judge_res['Redundant']:
                        image_dir = f"results_en_phase0_multi_init/7b_new_traj_{software}_maxtraj=15_r{r}/pyautogui/screenshot_a11y_tree/ui-tars/{software}/{init_id}/{i}"
                        all_image_files = glob.glob(os.path.join(image_dir, "step_*_*.png"))
                        print(len(all_image_files), init_id, i)
    print(cnt)
