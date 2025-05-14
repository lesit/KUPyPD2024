import os
import shutil
import argparse
import json
import time
import pandas as pd
from collections import defaultdict

import sys
module_path = os.path.abspath("../../")
if module_path not in sys.path:
    sys.path.append(module_path)

from kupypd_config import *
def check_best_hp_results(total_model_params_dict:dict, train_hp_search_dir):
    alternative_model_names = dict()
    for model_name, model_param_dict in total_model_params_dict.items():
        alter_model_name = model_param_dict.get("no_kc_alter")
        if alter_model_name is not None:
            alternative_model_names[alter_model_name] = model_name

    for dataset_name in os.listdir(train_hp_search_dir):
        dataset_result_dir = os.path.join(train_hp_search_dir, dataset_name)
        if not os.path.isdir(dataset_result_dir):
            continue

        hp_search_completed_models = defaultdict(list)
        hp_search_continuing_models = defaultdict(list)

        for model_name in sorted(os.listdir(dataset_result_dir)):
            model_dir_path = os.path.join(dataset_result_dir, model_name)
            if not os.path.isdir(model_dir_path):
                continue
            
            for fname in os.listdir(model_dir_path):
                file_path = os.path.join(model_dir_path, fname)
                if file_path.endswith("_train_results.json") and os.path.isfile(file_path):
                    with open(file_path, "r") as f:
                        valid_auc_dict = json.load(f)

                    hp_regular_dict = valid_auc_dict["hp_regular_dict"]
                    if "mean" in valid_auc_dict:
                        mean_auc = valid_auc_dict["mean"]
                        hp_search_completed_models[model_name].append([hp_regular_dict, mean_auc])
                    else:
                        hp_regular_str = hp_regular_to_str(hp_regular_dict)
                        hp_search_continuing_models[model_name].append(hp_regular_str)
                        
        models = '\n'.join([x for x in hp_search_continuing_models.keys()])
        print("continuing models:\n"+models+"\n")

        best_results_list = []
        for model_name, search_result_list in hp_search_completed_models.items():
            search_result_list = sorted(search_result_list, key=lambda x: x[1], reverse=True)

            hp_regular_dict, mean_auc = search_result_list[0]

            best_emb_size = hp_regular_dict[kupyd_hp_regular_key_list[0]]
            best_lr = hp_regular_dict[kupyd_hp_regular_key_list[1]]
            best_dropout = hp_regular_dict[kupyd_hp_regular_key_list[1]]
            
            best_emb_size = int(best_emb_size)
            best_lr = float(best_lr)
            best_dropout = float(best_dropout)

            best_valid_auc = float(f"{mean_auc:.4f}")

            hp_regular_str = hp_regular_to_str(hp_regular_dict)
            hp_search_dir = os.path.join(dataset_result_dir, model_name, hp_regular_str)
            assert os.path.isdir(hp_search_dir)

            model_param_dict = total_model_params_dict.get(model_name)
            if model_param_dict is None:
                org_model_name = alternative_model_names.get(model_name)
                if org_model_name is not None:
                    model_param_dict = total_model_params_dict[org_model_name]

            apply_hp_to_model_param_dict(hp_regular_dict, model_param_dict)

            best_results_list.append([model_name, len(search_result_list), best_valid_auc, best_emb_size, best_lr, best_dropout, hp_search_dir])
        
        df = pd.DataFrame(best_results_list, columns=["model_name", "total_hp_search", "valid_auc", "emb_size", "lr", "dropout", "train_saved_dir"])
        df.to_csv(os.path.join(dataset_result_dir, f"{dataset_name}_models_best_hp.csv"), index=False)

        save_new_model_params_json_fname = f"model_params.json"
        with open(os.path.join(dataset_result_dir, save_new_model_params_json_fname), "w") as f:
            json.dump(total_model_params_dict, f, indent=3)

def copy_hp_search_train_results(train_hp_search_dir, copy_to_dir):
    for dataset_name in os.listdir(train_hp_search_dir):
        dataset_result_dir = os.path.join(train_hp_search_dir, dataset_name)
        if not os.path.isdir(dataset_result_dir):
            continue

        hp_search_csv_path = os.path.join(dataset_result_dir, f"{dataset_name}_models_best_hp.csv")
        df = pd.read_csv(hp_search_csv_path)
        for idx, row in df.iterrows():
            model_name = row.model_name
            train_saved_dir = row.train_saved_dir
            model_copy_to_dir = os.path.join(copy_to_dir, dataset_name, model_name)

            if os.path.isdir(model_copy_to_dir):
                shutil.rmtree(model_copy_to_dir)

            shutil.copytree(train_saved_dir, model_copy_to_dir)

            saved_model_dir, hp_info = os.path.split(train_saved_dir)
            saved_result_json_path = os.path.join(saved_model_dir, f"{hp_info}_train_results.json")
            target_path = os.path.join(model_copy_to_dir, f"{model_name}_train_results.json")
            shutil.copy2(saved_result_json_path, target_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hp_search_dir", type=str)
    parser.add_argument("--model_param_jsons", type=str, default="model_params_default.json")
    parser.add_argument("--copy_results_dir", type=str, default="", help="destination to copy. just copy previous results")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    model_param_json_path_list = args.model_param_jsons.split(",")
    total_model_params_dict = dict()
    for model_param_json_path in model_param_json_path_list:
        with open(model_param_json_path, 'r') as f:
            model_params_dict = json.load(f)
            total_model_params_dict.update(model_params_dict)

    check_best_hp_results(total_model_params_dict, args.train_hp_search_dir)

    if len(args.copy_results_dir)>0:
        copy_hp_search_train_results(args.train_hp_search_dir, args.copy_results_dir)
