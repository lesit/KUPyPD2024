kupypd_train_hp_search_folder = "train_hp_search"
kupypd_train_folder = "train"
kupypd_eval_folder = "eval"

model_config_emb_size_name_list = ["emb_size", "d_model", "d"]
model_config_lr_name_list = ["learning_rate"]
model_config_dropout_name_list = ["dropout"]

kupyd_hp_regular_key_list = ["emb", "lr", "drop"]

def get_hp_regular(hyper_param_dict:dict):
    model_config_emb_size_names = set(model_config_emb_size_name_list)

    hp_regular = dict()
    for name, param in hyper_param_dict.items():
        if name in model_config_emb_size_names:
            name = kupyd_hp_regular_key_list[0]
        elif name in model_config_lr_name_list:
            name = kupyd_hp_regular_key_list[1]
        elif name in model_config_dropout_name_list:
            name = kupyd_hp_regular_key_list[2]

        hp_regular[name] = param
    
    return hp_regular

def hp_regular_to_str(hp_regular:dict):
    hp_list = [(key,hp_regular[key]) for key in kupyd_hp_regular_key_list]
    return '-'.join([f"{k}_{v}" for k,v in hp_list])

def str_to_hp_regular(hp_param_str):
    hp_param_str = hp_param_str.replace("1e-03", "0.001").replace("1e-04", "0.0001").replace("1e-05", "0.00001")
    for model_config_emb_size_name in model_config_emb_size_name_list:
        hp_param_str = hp_param_str.replace(model_config_emb_size_name+"_", kupyd_hp_regular_key_list[0]+"_")

    hp_param_str_list = hp_param_str.split("-")

    hyper_param_dict = dict()
    for value in hp_param_str_list:
        st = value.rfind("_")
        name = value[:st]
        value = value[st+1:]
        hyper_param_dict[name] = value

    return get_hp_regular(hyper_param_dict)

def get_hp_values(hp_regular_dict):
    emb_size = hp_regular_dict[kupyd_hp_regular_key_list[0]]
    lr = hp_regular_dict[kupyd_hp_regular_key_list[1]]
    dropout = hp_regular_dict[kupyd_hp_regular_key_list[2]]
    return emb_size, lr, dropout

def apply_hp_to_model_param_dict(hp_params:dict, to_model_params:dict):
    model_names_list = [model_config_emb_size_name_list, 
                      model_config_lr_name_list, 
                      model_config_dropout_name_list]
    
    for name_list, regular_key in zip(model_names_list, kupyd_hp_regular_key_list):
        for name in name_list:
            if name in to_model_params:
                to_model_params[name] = hp_params[regular_key]
                break
