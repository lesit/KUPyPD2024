import os
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emb_model", default="unixcoder")
parser.add_argument("--hidden", type=int, default=128)
args = parser.parse_args()

emb_model = args.emb_model

postfix = ""

result_dir = f"kupypd_result/search_hp_si_none_{emb_model}" + postfix

hp_check_list = {"lr":float, "batch_size":int}

including_hp_list = set()

semester_group_df_dict = dict()
for semester_group in sorted(os.listdir(result_dir)):
    semester_group_dir = os.path.join(result_dir, semester_group)

    if not os.path.isdir(semester_group_dir):
        continue

    df_list = []
    for fname in sorted(os.listdir(semester_group_dir)):
        if not fname.endswith("result.csv"):
            continue

        eval_result_path = os.path.join(semester_group_dir, fname)
        if not os.path.isfile(eval_result_path):
            continue

        df = pd.read_csv(eval_result_path)
        for hp_column in hp_check_list.keys():
            if hp_column in df.columns:
                including_hp_list.add(hp_column)

        df_list.append(df)

    if len(df_list)>0:
        semester_group_df_dict[semester_group] = df_list

import kupypd_config
default_config = kupypd_config.Config()

semeter_group_best_hp_dict = dict()
for semester_group in sorted(semester_group_df_dict.keys()):
    mean_list = []
    for df in semester_group_df_dict[semester_group]:
        mean_except_df = df[df["fold"] != "mean"]

        mean_row = mean_except_df.mean(numeric_only=True)
        mean_row["epoch"] = mean_row["epoch"].round()

        std_epochs = float(mean_except_df["epoch"].std())

        info = {"auc": mean_row.auc}

        hp_row = mean_except_df.iloc[0]

        for hp_column in including_hp_list:
            if hp_column in df.columns:
                v = hp_row[hp_column]
                type_cast = hp_check_list[hp_column]
                v = type_cast(v)
                info[hp_column] = v
            else:
                info[hp_column] = default_config.__getattribute__(hp_column)

        info.update({
            "epochs": int(mean_row.epoch),
            "std_epochs": std_epochs,
        })
        
        mean_list.append(info)

    mean_list = sorted(mean_list, key=lambda x: x["auc"], reverse=True)
    semeter_group_best_hp_dict[semester_group] = mean_list[0]

save_path = os.path.join(result_dir, "semester_group_best_hp.json")
with open(save_path, "w") as f:
    json.dump(semeter_group_best_hp_dict, f, indent=3)

import shutil
shutil.copy2(save_path, f"semester_group_best_hp_{emb_model}{postfix}.json")
