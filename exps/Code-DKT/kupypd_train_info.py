import os
import json
import copy
import kupypd_config

def make_train_list(default_config:kupypd_config.Config, dataset_hp_path, hp_comb_list_or_dict, semester_group_dir_list, data_aux_root_dir, train_save_dir):
    dataset_hp_dict = None
    if dataset_hp_path is not None:
        with open(dataset_hp_path, "r") as f:
            dataset_hp_dict = json.load(f)

    overall_except_fields = {"auc", "std_epochs"}

    def set_config_from_hp_dict(semester_group, config, excepts:set=None):
        if dataset_hp_dict is not None:
            dataset_hp = dataset_hp_dict[semester_group]
            prefix = ""

            config_except_fields = excepts.union(overall_except_fields)
            for k, v in dataset_hp.items():
                if k not in config_except_fields:
                    config.__setattr__(k, v)
                    prefix += f"{k}_{v}-"

            if len(prefix) > 0:
                return prefix

        return "default_cfg-"

    train_info_list = []
    if hp_comb_list_or_dict is not None:
        for semester_group, data_dir in semester_group_dir_list:
            if isinstance(hp_comb_list_or_dict, list):
                hp_comb_list = hp_comb_list_or_dict
            elif isinstance(hp_comb_list_or_dict, dict):
                hp_comb_list = hp_comb_list_or_dict[semester_group]
            else:
                assert False, "hp_comb_list_or_dict must be list or dict"

            if semester_group == '20241_1':
                a = 0

            for hp_comb in hp_comb_list:
                config = copy.deepcopy(default_config)

                for k, v in hp_comb.items():
                    config.__setattr__(k, v)

                prefix = set_config_from_hp_dict(semester_group, config, set(hp_comb.keys()))

                def make_name(kv_sep:str, name_sep:str):
                    replace_hp_name = {"abandon_low_frequence": "a_low_freq"}
                    return prefix + name_sep.join([f"{replace_hp_name.get(k,k)}{kv_sep}{v}" for k, v in hp_comb.items()])

                hp_name = make_name(":", ",")
                result_name = make_name("_", "-")

                train_info_list.append({
                    "semester_group": semester_group,
                    "data_dir": data_dir, 
                    "data_aux_path": os.path.join(data_aux_root_dir, f"{semester_group}_submission_aux.csv") if data_aux_root_dir is not None else None,
                    "config": config,
                    "hp_comb": hp_comb,
                    "unique_key": f"{semester_group}_{hp_name}",
                    "fold_result_dir": os.path.join(train_save_dir, semester_group, result_name),
                    "save_result_path": os.path.join(train_save_dir, semester_group, result_name+"-result.csv")
                    })
    else:
        for semester_group, data_dir in semester_group_dir_list:
            config = copy.deepcopy(default_config) 
            set_config_from_hp_dict(semester_group, config)

            train_info_list.append({
                "semester_group": semester_group,
                "data_dir": data_dir, 
                "data_aux_path": os.path.join(data_aux_root_dir, f"{semester_group}_data_aux.csv") if data_aux_root_dir is not None else None,
                "config": config,
                "unique_key": semester_group,
                "fold_result_dir": os.path.join(train_save_dir, semester_group),
                "save_result_path": os.path.join(train_save_dir, f"{semester_group}.csv")
                })
    for idx, train_info in enumerate(train_info_list):
        train_info["log_prefix"] = f"{idx+1}/{len(train_info_list)}." + train_info["unique_key"]
    return train_info_list
