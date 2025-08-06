import json
import os

# similar to AgentRewardBench, we use precision and NPV to evaluate the performance of the judge model

TP, TN, FP, FN = 0, 0, 0, 0
all_episodes = json.load(open("evaluation_examples/test_all.json", "r"))
mid_pth = '72b_multi_env=8_xxx' # mid_pth of trajectory.
for software in all_episodes.keys():
    software_pth  = software
    if software == 'vs_code':
        software_pth = 'vscode'
    for episode_id in all_episodes[software]:
        instruction_file = f"evaluation_examples/examples/{software}/{episode_id}.json"
        with open(instruction_file, "r", encoding="utf-8") as f:
            task = json.load(f)
        if task['evaluator']['func'] == 'infeasible':
            continue
        res_dir = f"results_en/{mid_pth}/pyautogui/screenshot_a11y_tree/ui_tars_72b_dpo/{software}/{episode_id}/"
        if not os.path.exists(f"{res_dir}/result.txt"):
            continue
        gt_success = False
        with open(f"{res_dir}/result.txt", 'r') as f:
            res = f.readlines()[0]
            if '1' in res:
                gt_success = True
            else:
                gt_success = False
        try:
            # Where the judge model prediction is stored after `judge_full_process`.
            pred = json.load(open(f'judge_qwen2.5_vl_7b_{software_pth}_ori/72b_multi_env=8_t=0.5_{software_pth}_r0_w_res/{episode_id}.json'))
            pred_success = pred['Correctness']
        except:
            pred_success = False
        if gt_success and pred_success:
            TP += 1
        elif not gt_success and not pred_success:
            TN += 1
        elif not gt_success and pred_success:
            FP += 1
        elif gt_success and not pred_success:
            FN += 1

print("\n=== Evaluation Results ===")
print(f"True Positive (TP): {TP}")
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0

print(f"\nPrecision: {precision * 100:.2f}%")
print(f"NPV: {npv * 100:.2f}%")
