import os
import csv

def get_result(action_space, use_model, observation_type, result_dir, summary_csv_path="summary_success_rate_1.5.csv"):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    domain_result = {}
    all_result_for_analysis = {}

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path) and "result.txt" in os.listdir(example_path):
                    if domain not in domain_result:
                        domain_result[domain] = []
                    try:
                        result_str = open(os.path.join(example_path, "result.txt"), "r").read()
                        result_val = float(result_str)
                    except:
                        result_val = float(eval(result_str))
                    domain_result[domain].append(result_val)

                    if domain not in all_result_for_analysis:
                        all_result_for_analysis[domain] = {}
                    all_result_for_analysis[domain][example_id] = result_val

                    all_result.append(result_val)

    # 保存 JSON 结果分析
    with open(os.path.join(target_dir, "all_result.json"), "w") as f:
        f.write(str(all_result_for_analysis))

    # 所有 domain 统一排序，确保列对齐
    all_domains = sorted(domain_result.keys())

    count_row = [result_dir]
    sr_row = [result_dir]
    total_count = 0
    sr_sum = 0.0
    domain_sr_count = 0

    for domain in all_domains:
        count = len(domain_result[domain])
        sr = round(sum(domain_result[domain]) / count * 100, 2) if count > 0 else 0.0
        count_row.append(count)
        sr_row.append(sr)
        total_count += count
        sr_sum += sr
        domain_sr_count += 1

    count_row.extend([total_count, "-"])
    mean_sr = round(sr_sum / domain_sr_count, 2) if domain_sr_count > 0 else 0.0
    sr_row.extend(["-", mean_sr])

    # 写入 CSV：若不存在则写表头，存在则追加
    file_exists = os.path.exists(summary_csv_path)
    with open(summary_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            header = ["results_dir"] + [f"{d}" for d in all_domains] + ["cnt_all", "mean_SR(%)"]
            writer.writerow(header)
        writer.writerow(count_row)
        writer.writerow(sr_row)

    print(f"Summary appended to {summary_csv_path}")
    print("Total count:", total_count, "Mean SR(%):", mean_sr)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Runned:", len(all_result), "Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == '__main__':
    get_result("pyautogui", "ui_tars_1.5_7b", "screenshot_a11y_tree", f"results_en_test_1.5_fullosworld/all_l15_h1/UI-TARS-1.5-7B")
