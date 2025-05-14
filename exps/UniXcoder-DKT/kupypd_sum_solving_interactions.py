import os
import json
import pandas as pd
import argparse

import kupypd_config

parser = argparse.ArgumentParser()
parser.add_argument("--emb_model", default="unixcoder")
parser.add_argument('--solving_interactions', 
                    choices=[x.name for x in kupypd_config.solving_interaction_types], 
                    default=kupypd_config.solving_interaction_types.none.name)

args = parser.parse_args()

emb_model = args.emb_model

result_dir = f"kupypd_result/train_eval_si_{args.solving_interactions}_{emb_model}"

with open(os.path.join(f"semester_group_best_hp_{emb_model}.json"), "r") as f:
    dataset_hp_dict = json.load(f)

from collections import defaultdict
semester_group_list = []
semester_group_results = defaultdict(list)

for semester_group in sorted(os.listdir(result_dir)):
    semester_group_dir = os.path.join(result_dir, semester_group)
    if not os.path.isdir(semester_group_dir):
        continue

    semester_group_hp_dict = dataset_hp_dict[semester_group]

    result_df_list = []
    for fname in sorted(os.listdir(semester_group_dir)):
        if not fname.endswith("-result.csv"):
            continue

        eval_result_path = os.path.join(semester_group_dir, fname)
        if not os.path.isfile(eval_result_path):
            continue

        df = pd.read_csv(eval_result_path)

        mean_rows = df[df["fold"] == "epoch_mean"]
        mean_rows = mean_rows.drop(columns=["fold"])

        solving_interaction_len = df["solving_interaction_len"].iloc[0]
        epochs = semester_group_hp_dict["epochs"]

        mean_row = mean_rows[mean_rows["epoch"] == epochs]
        result_df_list.append(mean_row)

    result_df = pd.concat(result_df_list, ignore_index=True)
    if "epochs" in result_df.columns:
        result_df.pop("epochs")
    result_df.insert(0, "solving_interaction_len", result_df.pop("solving_interaction_len"))
    result_df = result_df.sort_values(by=["solving_interaction_len", "emb_solving_interaction_size"])
    result_df.to_csv(os.path.join(result_dir, f"{semester_group}_summary.csv"), index=False)

    semester_group_list.append(semester_group)
    for idx, row in result_df.iterrows():
        key = (int(row.solving_interaction_len), int(row.emb_solving_interaction_size))
        semester_group_results[key].append(row.auc)

row_list = []
hp_list = sorted(semester_group_results.keys())
for hp in hp_list:
    row = list(hp)
    row += semester_group_results[hp]
    row_list.append(row)

frequency_df = pd.DataFrame(row_list, columns=["interaction len", "embedding size"] + semester_group_list)
frequency_df.to_csv(os.path.join(result_dir, f"summary_{emb_model}.csv"), index=False)

