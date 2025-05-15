# copied main function from examples/wandb_train.py
# and, modified a little

import os
import shutil
import argparse
import json
import time
import sys
module_path = os.path.abspath("../../")
if module_path not in sys.path:
	sys.path.append(module_path)

import src.make_logger as make_logger

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout, ensure_ascii=False, indent=3)

config_dir = "configs"

def kupypd_train(params, logger, log_prefix=""):
    import torch
    torch.set_num_threads(4) 

    from torch.optim import SGD, Adam
    import copy

    from pykt.models import train_model as pykt_train_model
    from pykt.models import init_model as pykt_init_model
    from pykt.models import load_model as pykt_load_model
    from pykt.models import evaluate as pykt_evaluate
    from pykt.utils import debug_print,set_seed
    from pykt.datasets import init_dataset4train
    import time

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    # if "use_wandb" not in params:
    #     params['use_wandb'] = 1

    # if params['use_wandb']==1:
    #     import wandb
    #     wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]

    s = time.time()
    logger.info(f"{log_prefix} kupypd_train. start. dataset_name:{dataset_name}. fold:{fold}")

    logger.info(f"{log_prefix} kupypd_train. load config files.")
    with open(os.path.join(config_dir, "kt_config.json")) as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt","folibikt", "atkt", "lpkt", "skvmn", "dimkt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["simplekt","stablekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32 ## because of OOM

        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']

    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open(os.path.join(config_dir, "data_config.json")) as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    logger.info(f"{log_prefix} kupypd_train. Start init data. batch_size:{batch_size}")
    
    logger.info(f"{log_prefix} kupypd_train. init_dataset")
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level)

    logger.info(f"{log_prefix} Start training model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}, fold: {fold}")

    model_config = copy.deepcopy(params)
    for remove_item in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed", 'use_wandb','learning_rate','batch_size','add_uuid','l2', 'num_epochs']:
        if remove_item in model_config:
            del model_config[remove_item]

    if model_name in ["dimkt"]:
        del model_config['weight_decay']

    if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt","stablekt","sparsekt", "bakt_time","folibikt"]:
        model_config["seq_len"] = seq_len

    logger.info(f"{log_prefix} model_config: {model_config}")
    logger.info(f"{log_prefix} train_config: {train_config}")

    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1

    ckpt_path = save_dir

    if os.path.isdir(ckpt_path):
        shutil.rmtree(ckpt_path)

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
        
    logger.info(f"{log_prefix} pykt_init_model")
    model = pykt_init_model(model_name, model_config, data_config[dataset_name], emb_type)
    # logger.info(f"model is {model}")

    learning_rate = params["learning_rate"]
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        logger.info(f"{log_prefix} dtransformer weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

    logger.info(f"{log_prefix} train model :")

    save_model = True
    if model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = \
            pykt_train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, data_config[dataset_name], fold, logger=logger)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = pykt_train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, logger=logger)

    torch.cuda.empty_cache()

    logger.info(f"{log_prefix} kupypd_train. end. valid_auc:{validauc}, valid_acc:{validacc}. best_epoch:{best_epoch} elapse: {time.time() - s}")
    return validauc
    
import pandas as pd

def train_folds(param_dict:dict, save_dir, train_results_path, log_prefix, logger, debug=False, **kwargs):
    dataset_name = param_dict["dataset_name"]

    logger.info(f"{log_prefix} start")

    with open(os.path.join(config_dir, "data_config.json")) as fin:
        data_config = json.load(fin)
    fold_list = data_config[dataset_name]["folds"]
    if debug:
        fold_list = fold_list[:1]

    st = time.time()

    logger.info(f"result_path: {train_results_path}")
    valid_auc_dict = dict()
    if os.path.isfile(train_results_path):
        with open(train_results_path, "r") as f:
            valid_auc_dict = json.load(f)

    if "hp_regular_dict" in kwargs:
        valid_auc_dict["hp_regular_dict"] = kwargs["hp_regular_dict"]

    valid_auc_list = []
    for fold in fold_list:
        fold_key = f"fold_{fold}"
        fold_log_prefix = f"{log_prefix}. fold:{fold}"

        if fold_key in valid_auc_dict:
            valid_auc_list.append(valid_auc_dict[fold_key])
            logger.info(f"{fold_log_prefix}. aleady trained")
            continue

        param_dict["fold"] = fold
        param_dict["save_dir"] = os.path.join(save_dir, fold_key)

        valid_auc = kupypd_train(param_dict, logger, fold_log_prefix)
        valid_auc_dict[fold_key] = valid_auc

        with open(train_results_path, "w") as f:
            json.dump(valid_auc_dict, f, indent=3)

        valid_auc_list.append(valid_auc)
        logger.info(f"\n")

    mean_auc = sum(valid_auc_list) / len(valid_auc_list)
    valid_auc_dict["mean"] = mean_auc
    with open(train_results_path, "w") as f:
        json.dump(valid_auc_dict, f, indent=3)

    logger.info(f"{log_prefix} end. elapse:{time.time()-st:.2f} end")

    return valid_auc_dict

from kupypd_config import *

def make_hp_search_train_list(save_root_dir, train_param_dict:dict, hp_search_comb_list:list, unique_key:str):
    model_name = train_param_dict["model_name"]
    dataset_name = train_param_dict["dataset_name"]

    model_save_dir = os.path.join(save_root_dir, dataset_name, model_name)

    train_list = []
    if hp_search_comb_list is not None:
        for idx, hyper_param_dict in enumerate(hp_search_comb_list):
            hp_regular_dict = get_hp_regular(hyper_param_dict)
            hp_regular_str = hp_regular_to_str(hp_regular_dict)

            hp_search_train_param_dict = train_param_dict.copy()
            emb_size, lr, dropout = get_hp_values(hp_regular_dict)
            for key in hp_search_train_param_dict.keys():
                if key in model_config_emb_size_name_list:
                    hp_search_train_param_dict[key] = emb_size
                elif key == "learning_rate":
                    hp_search_train_param_dict[key] = lr
                elif key == "dropout":
                    hp_search_train_param_dict[key] = dropout

            hp_search_model_save_dir = os.path.join(model_save_dir, hp_regular_str)
            train_results_fname = f"{hp_regular_str}_train_results.json"
            log_prefix = f"model:{model_name}. data:{dataset_name}. {idx+1}/{len(hp_search_comb_list)}. param test:{hp_regular_str}."

            train_list.append({
                "param_dict": hp_search_train_param_dict,
                "save_dir": hp_search_model_save_dir,
                "train_results_path": os.path.join(model_save_dir, train_results_fname),
                "log_prefix": log_prefix,
                "unique_key": f"{unique_key}_hp:{hp_regular_str}",
                "hp_regular_dict": hp_regular_dict
                })
    else:
        train_results_fname = f"{model_name}_train_results.json"
        log_prefix = f"model:{model_name}. data:{dataset_name}."

        train_list.append({
            "param_dict": train_param_dict,
            "save_dir": model_save_dir,
            "train_results_path": os.path.join(model_save_dir, train_results_fname),
            "log_prefix": log_prefix,
            "unique_key": unique_key,
            })
    return train_list

def make_all_train_list(dataset_names, dataset_total_model_params_dict, hp_search_comb_list, train_save_dir):
    train_list = []
    for dataset_name in dataset_names:
        total_model_params_dict = dataset_total_model_params_dict[dataset_name]
        for model_name, param_dict in total_model_params_dict.items():
            train_param_dict = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "use_wandb": 0,
                "add_uuid": 0,
            }
        
            train_param_dict.update(param_dict)

            if args.debug:
                train_param_dict["num_epochs"] = 1      ##### debugging

            hp_search_train_list = make_hp_search_train_list(train_save_dir, train_param_dict, hp_search_comb_list,
                                                             unique_key=f"d:{dataset_name}_m:{model_name}")
            train_list.extend(hp_search_train_list)
    for idx, train_info in enumerate(train_list):
        train_info["log_prefix"] = f"{idx+1}/{len(train_list)}. " + train_info["log_prefix"]
    return train_list

from src.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1", help="gpu devices")
    parser.add_argument("--model_param_jsons_fname", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default="20241_0,20241_1,20241_2,20241_3,20242_0")
    parser.add_argument("--hp_tunes", type=str, default=None)
    parser.add_argument("--inc_accepted_reattempt", action="store_true")
    parser.add_argument("--result_dir", type=str, default="kupypd_result")
    parser.add_argument("--log_dir", type=str, default="kupypd_log")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    save_new_model_params_json_fname=f"model_params.json"

    model_param_jsons_fname = args.model_param_jsons_fname
    if model_param_jsons_fname is None:
        model_param_jsons_fname = save_new_model_params_json_fname

    hp_search_comb_list = None
    if args.hp_tunes is not None:
        with open(args.hp_tunes, 'r') as f:
            hp_scopes = json.load(f)

        if len(hp_scopes)>0:
            hp_search_comb_list = make_hp_combs(hp_scopes)
            if len(hp_search_comb_list) == 0:
                hp_search_comb_list = None

    def get_save_dir(save_root_dir, is_hp_search):
        if is_hp_search:
            return os.path.join(save_root_dir, kupypd_train_hp_search_folder)
        else:
            return os.path.join(save_root_dir, kupypd_train_folder)

    train_save_dir = get_save_dir(args.result_dir, hp_search_comb_list is not None)
    if args.inc_accepted_reattempt:
        train_save_dir += "_inc_accepted_reattempt"

    logger_dir = get_save_dir(args.log_dir, hp_search_comb_list is not None)

    logger_name = f"kupypd_train-mp_{model_param_jsons_fname}-device_{args.devices}"
    if args.inc_accepted_reattempt:
        logger_name += "_inc_accepted_reattempt"
    if hp_search_comb_list is not None:
        logger_name += f"-hp_{args.hp_tunes}"

    import datetime
    logger_name += '.' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    logger = make_logger.make(logger_name, time_filename=False, save_dir=logger_dir)

    arg_params = vars(args)
    logger.info(f"args:\n{arg_params}")

    dataset_names = [dataset_name for dataset_name in args.dataset_names.split(",")]
    if args.inc_accepted_reattempt:
        dataset_names = [dataset_name+"_inc_accepted_reattempt" for dataset_name in dataset_names]

    with open(os.path.join("pykt", "config", "que_type_models.json"), "r") as f:
        que_type_model_info = json.load(f)
        que_type_models = que_type_model_info["que_type_models"]
        qikt_ab_models = que_type_model_info["qikt_ab_models"]
        que_type_models += qikt_ab_models

    que_type_models = set(que_type_models)

    def load_total_model_params_dict(model_params_dir):
        with open(os.path.join(model_params_dir, model_param_jsons_fname), 'r') as f:
            total_model_params_dict = json.load(f)
        refined_model_params_dict = dict()
        for model_name, model_params_dict in total_model_params_dict.items():
            if model_name in que_type_models:
                logger.info(f"no kc: {model_name} is not available: que_type_models")
                continue

            no_kc_available = model_params_dict.get("no_kc_available", True)
            if not no_kc_available:
                logger.info(f"no kc: {model_name} is not available")
                continue

            alternative_model = model_params_dict.get("no_kc_alter")
            if alternative_model is not None:
                logger.info(f"no kc: {model_name} --> {alternative_model}")

                model_name = alternative_model
            
            refined_model_params_dict[model_name] = model_params_dict
        return refined_model_params_dict

    dataset_total_model_params_dict = dict()
    if args.hp_tunes is not None:
        total_model_params_dict = load_total_model_params_dict(".")
        for dataset_name in dataset_names:
            dataset_total_model_params_dict[dataset_name] = total_model_params_dict
    else:
        hp_search_save_dir = get_save_dir(args.result_dir, True)
        if args.inc_accepted_reattempt:
            hp_search_save_dir += "_inc_accepted_reattempt"
        for dataset_name in dataset_names:
            dataset_model_params_dir = os.path.join(hp_search_save_dir, dataset_name)
            total_model_params_dict = load_total_model_params_dict(dataset_model_params_dir)
            dataset_total_model_params_dict[dataset_name] = total_model_params_dict

    train_info_list = make_all_train_list(dataset_names, dataset_total_model_params_dict, hp_search_comb_list, train_save_dir)

    st = time.time()

    gpu_devices = args.devices.split(",")
    if len(gpu_devices) < 2 or args.debug:
        logger.info("train models with single core")

        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

        trained_list = []
        for train_info in train_info_list:
            try:
                train_folds(**train_info, logger=logger, debug=args.debug)
            except Exception as e:
                make_logger.write_exception_log(logger, e)
                continue

            trained_list.append(train_info["unique_key"])
            logger.info("")

        logger.info(f"trained list: total:{len(trained_list)}\n" + "\n".join(trained_list))
    else:
        logger.info(f"train models with multiprocessing: gpus:{gpu_devices}. start")

        import multiprocessing

        shared_lock = multiprocessing.Lock()
        manager = multiprocessing.Manager()
        shared_check_dict = manager.dict()

        def train_process(shared_lock, logger_name, gpu, train_info_list, shared_check_dict, debug):
            mp_logger = make_logger.make(f"{logger_name}_gpu{gpu}", time_filename=False, save_dir=logger_dir)
            mp_logger.info(f"train models with multiprocessing: gpus:{gpu}. start")

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu

            trained_list = []
            for train_info in train_info_list:
                shared_lock.acquire()
                check_key = train_info["unique_key"]
                if check_key in shared_check_dict:
                    log_prefix = train_info["log_prefix"]
                    mp_logger.info(f"{log_prefix}. already started in other process\n")
                    shared_lock.release()
                    continue

                shared_check_dict[check_key] = True
                shared_lock.release()

                try:
                    train_folds(**train_info, logger=mp_logger, debug=debug)
                except Exception as e:
                    make_logger.write_exception_log(mp_logger, e)
                    exit(-1)

                trained_list.append(check_key)
                mp_logger.info("")

            mp_logger.info(f"trained list: total:{len(trained_list)}\n" + "\n".join(trained_list))
            mp_logger.info(f"train models with multiprocessing: gpus:{gpu}. end")

        process_list = []
        for gpu in gpu_devices:
            process = multiprocessing.Process(target=train_process, args=[shared_lock, logger_name, gpu, train_info_list, shared_check_dict, args.debug])
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

        logger.info(f"train models with multiprocessing: gpus:{gpu_devices}. end")

    if hp_search_comb_list is not None:
        from kupypd_check_best_hp_results import check_best_hp_results, copy_hp_search_train_results

        check_best_hp_results(total_model_params_dict, train_save_dir)

        copy_to_dir = get_save_dir(args.result_dir, is_hp_search = False)
        if args.inc_accepted_reattempt:
            copy_to_dir += "_inc_accepted_reattempt"
        copy_hp_search_train_results(train_save_dir, copy_to_dir)

    logger.info(f"completed!!!. total elapse:{time.time()-st}")
