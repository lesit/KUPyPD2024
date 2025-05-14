import os
import json
import pandas as pd

# postfix = "_inc_accepted_reattempt"
postfix = ""

result_dir = "kupypd_result/train_ablation_study" + postfix

with open(os.path.join("semester_group_best_hp.json"), "r") as f:
    dataset_hp_dict = json.load(f)

from collections import defaultdict

semester_group_list = []
result_list = []
n_freq = 4
for semester_group in sorted(os.listdir(result_dir)):
    semester_group_dir = os.path.join(result_dir, semester_group)
    if not os.path.isdir(semester_group_dir):
        continue

    semester_group_hp_dict = dataset_hp_dict[semester_group]

    semester_group_result_df_list = []
    for fname in sorted(os.listdir(semester_group_dir)):
        if not fname.endswith("-result.csv"):
            continue

        eval_result_path = os.path.join(semester_group_dir, fname)
        if not os.path.isfile(eval_result_path):
            continue

        df = pd.read_csv(eval_result_path)

        mean_rows = df[df["fold"] == "epoch_mean"]
        mean_rows = mean_rows.drop(columns=["fold"])
    
        epochs = semester_group_hp_dict["epochs"]
        mean_row = mean_rows[mean_rows["epoch"] == epochs]
        semester_group_result_df_list.append(mean_row)

    result_df = pd.concat(semester_group_result_df_list, ignore_index=True)
    result_df = result_df.sort_values(by=["abandon_low_frequence"])
    result_df.to_csv(os.path.join(result_dir, f"{semester_group}_summary.csv"), index=False)

    semester_group_results = []
    for idx, row in result_df.iterrows():
        semester_group_results.append([int(row.abandon_low_frequence), row.auc])

    n_freq = len(semester_group_results)
    semester_group_list.append(semester_group)
    result_list.append(semester_group_results)

row_list = []
for idx, ratio in enumerate(["0%", "1%", "5%", "10%"]):
    row = [ratio] + [x[idx][0] for x in result_list]
    row_list.append(row)

frequency_df = pd.DataFrame(row_list, columns=["frequency"] + semester_group_list)
frequency_df.to_csv(os.path.join(result_dir, f"frequency.csv"), index=False)

row_list = []
for idx, ratio in enumerate(["0%", "1%", "5%", "10%"]):
    row = [ratio] + [x[idx][1] for x in result_list]
    row_list.append(row)
frequency_df = pd.DataFrame(row_list, columns=["remove low frequency"] + semester_group_list)
frequency_df.to_csv(os.path.join(result_dir, f"summary.csv"), index=False)

