import os
import argparse
import json
import copy
import pandas as pd
import numpy as np
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

config_dir = "configs"

def main(arg_params, data_train_fold_dir, fold_log_prefix, logger):
    logger.info(f"{fold_log_prefix}. eval start")

    import torch
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    from pykt.models import evaluate,evaluate_question,load_model
    from pykt.datasets import init_test_datasets

    fusion_type = arg_params["fusion_type"].split(",")

    with open(os.path.join(data_train_fold_dir, "config.json")) as fin:
        config = json.load(fin)
        batch_size = config["train_config"]["batch_size"]
        
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]    
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len   

    with open(os.path.join(config_dir, "data_config.json")) as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]    

    if model_name not in ["dimkt"]:        
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    logger.info(f"{fold_log_prefix}.Start predicting model: {model_name}, embtype: {emb_type}, saved_dir: {data_train_fold_dir}, dataset_name: {dataset_name}")
    logger.info(f"{fold_log_prefix}.model_config: {model_config}")
    logger.info(f"{fold_log_prefix}.data_config: {data_config}")

    model = load_model(model_name, model_config, data_config, emb_type, data_train_fold_dir)

    save_test_path = os.path.join(data_train_fold_dir, model.emb_type+"_test_predictions.txt")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))                

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path)
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    logger.info(f"{fold_log_prefix}. testauc: {testauc}, testacc: {testacc}")

    torch.cuda.empty_cache()

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(data_train_fold_dir, model.emb_type+"_test_window_predictions.txt")
    if model.model_name == "rkt":
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_test_window_path)
    else:
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path)
    logger.info(f"{fold_log_prefix}. window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    torch.cuda.empty_cache()

    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    q_testaucs, q_testaccs = -1,-1
    qw_testaucs, qw_testaccs = -1,-1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(data_train_fold_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]

        torch.cuda.empty_cache()

    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(data_train_fold_dir, model.emb_type+"_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]

        torch.cuda.empty_cache()

    logger.info(f"{fold_log_prefix}. eval end. elapse: {time.time()-st}\n{dres}")
    return dres

def eval_folds(arg_params, log_prefix, dataset_name, model_name, saved_dir, eval_result_path, logger, **kwargs):
    if os.path.isfile(eval_result_path):
        logger.info(f"{log_prefix}. evaluated already")
        return

    st = time.time()
    logger.info(f"{log_prefix}. eval_folds.start")

    df_list = []
    for fold in sorted(os.listdir(saved_dir)):
        data_train_fold_dir = os.path.join(saved_dir, fold)
        if not os.path.isdir(data_train_fold_dir):
            continue

        fold_log_prefix = f"{log_prefix}. fold:{fold}"
        dres = main(arg_params, data_train_fold_dir, fold_log_prefix, logger)
        
        values = [[float(f"{float(x):.4f}") for x in dres.values()]]
        df = pd.DataFrame(data=values, columns=list(dres.keys()))
        df_list.append(df)

    total_df = pd.concat(df_list, ignore_index=True)
            
    performance_mean = np.mean(total_df.values,axis=0)
    performance_mean = [float(f"{float(x):.4f}") for x in performance_mean]

    performances = []
    performances.append(["mean"] + performance_mean)
    for fold, performance in enumerate(total_df.values):
        performances.append([str(fold)] + list(performance))
                
    eval_df = pd.DataFrame(performances, columns=['fold'] + list(total_df.columns))
    eval_df.insert(0, "dataset", dataset_name)
    eval_df.insert(1, "model", model_name)

    target_dir = os.path.split(eval_result_path)[0]
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    eval_df.to_csv(eval_result_path , index=False)
    logger.info(f"{log_prefix}. eval_folds.end. elapse: {time.time()-st}")

import sys
module_path = os.path.abspath("../../")
if module_path not in sys.path:
	sys.path.append(module_path)

import src.make_logger as make_logger

from kupypd_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1", help="gpu devices")
    parser.add_argument("--mp_per_gpu", default=2, type=int, help="multi processes per a gpu")
    parser.add_argument("--_inc_accepted_reattempt", action="store_true")
    parser.add_argument("--train_saved_dir", type=str, default="kupypd_result/train")
    parser.add_argument("--result_dir", type=str, default="kupypd_result")
    parser.add_argument("--log_dir", type=str, default="kupypd_log")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    train_saved_dir = args.train_saved_dir
    if args._inc_accepted_reattempt:
        train_saved_dir += "_inc_accepted_reattempt"
    train_saved_folder = os.path.split(train_saved_dir)[-1]

    logger_name = f"kupypd_eval-device_{args.devices}-{train_saved_folder}"
    logger_dir = os.path.join(args.log_dir, kupypd_eval_folder)

    import datetime
    logger_name += '.' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    logger = make_logger.make(logger_name, time_filename=False, save_dir=logger_dir)

    arg_params = vars(args)
    logger.info(f"args:\n{arg_params}")

    eval_save_dir = os.path.join(args.result_dir, kupypd_eval_folder)
    if args._inc_accepted_reattempt:
        eval_save_dir += "_inc_accepted_reattempt"
   
    eval_info_list = []
    for dataset_name in sorted(os.listdir(train_saved_dir)):
        dataset_trained_dir = os.path.join(train_saved_dir, dataset_name)
        if not os.path.isdir(dataset_trained_dir):
            continue

        for model_name in sorted(os.listdir(dataset_trained_dir)):
            saved_model_dir = os.path.join(dataset_trained_dir, model_name)
            # if model_name.startswith("/") or not os.path.isdir(saved_model_dir):
            #     continue
            if not os.path.isdir(saved_model_dir):
                continue

            if not os.path.isfile(os.path.join(saved_model_dir, f"{model_name}_train_results.json")):
                logger.info(f"dataset:{dataset_name}. model:{model_name} is not trained")
                continue

            eval_result_path = os.path.join(eval_save_dir, dataset_name, f"{model_name}_eval_results.csv")

            eval_info_list.append({
                "dataset_name": dataset_name,
                "model_name": model_name,
                "saved_dir": saved_model_dir,
                "eval_result_path": eval_result_path,
                "unique_key": f"d:{dataset_name}_m:{model_name}"
                })

    for idx, eval_info in enumerate(eval_info_list):
        eval_info["log_prefix"] = f"{idx+1}/{len(eval_info_list)}_" + eval_info["unique_key"]

    st = time.time()

    gpu_devices = []
    for idx in range(args.mp_per_gpu):
        for gpu in args.devices.split(","):
            gpu_devices.append(gpu)

    if len(gpu_devices) < 2 or args.debug:
        logger.info("eval models with single core")

        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

        for eval_info in eval_info_list:
            try:
                eval_folds(arg_params=arg_params, **eval_info, logger=logger)
            except Exception as e:
                make_logger.write_exception_log(logger, e)
                continue
            logger.info("")

        logger.info(f"eval list: total:{len(eval_info_list)}\n" + "\n".join(eval_info_list))
    else:
        logger.info(f"eval models with multiprocessing: gpus:{gpu_devices}. start")

        import multiprocessing

        shared_lock = multiprocessing.Lock()
        manager = multiprocessing.Manager()
        shared_check_dict = manager.dict()

        def eval_process(arg_params, logger_name, logger_dir, gpu, eval_info_list, shared_check_dict):
            mp_logger = make_logger.make(f"{logger_name}_gpu{gpu}", time_filename=False, save_dir=logger_dir)
            mp_logger.info(f"eval models with multiprocessing: gpus:{gpu}. start")

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu

            evaluated_list = []
            for eval_info in eval_info_list:
                shared_lock.acquire()
                check_key = eval_info["unique_key"]
                if check_key in shared_check_dict:
                    mp_logger.info(f"{check_key}. already started")
                    shared_lock.release()
                    continue
                else:
                    shared_check_dict[check_key] = True
                    shared_lock.release()

                    try:
                        eval_folds(arg_params=arg_params, **eval_info, logger=mp_logger)
                    except Exception as e:
                        make_logger.write_exception_log(mp_logger, e)
                        exit(-1)

                    evaluated_list.append(check_key)
                mp_logger.info("")

            mp_logger.info(f"evaluated_list list: total:{len(evaluated_list)}\n" + "\n".join(evaluated_list))
            mp_logger.info(f"eval models with multiprocessing: gpus:{gpu}. end")

        process_list = []
        for idx, gpu in enumerate(gpu_devices):
            process = multiprocessing.Process(target=eval_process, args=[arg_params, 
                                                                         f"{logger_name}_mp{idx}", 
                                                                         logger_dir, 
                                                                         gpu, 
                                                                         eval_info_list, 
                                                                         shared_check_dict])
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

        logger.info(f"eval models with multiprocessing: gpus:{gpu_devices}. end")

    from collections import defaultdict
    dataset_eval_results_dict = defaultdict(list)
    for eval_info in eval_info_list:
        dataset_name = eval_info["dataset_name"]
        eval_result_path = eval_info["eval_result_path"]

        df = pd.read_csv(eval_result_path)
        df = df[df.fold == "mean"]
        df.drop(columns=["fold"])
        dataset_eval_results_dict[dataset_name].append(df)

    model_eval_list_dict = defaultdict(list)
    dataset_name_list =[]
    for dataset_name in sorted(dataset_eval_results_dict.keys()):
        dataset_eval_df = pd.concat(dataset_eval_results_dict[dataset_name])
        dataset_eval_df.to_csv(os.path.join(eval_save_dir, f"{dataset_name}_eval_results.csv"), index=False) 

        dataset_name_list.append(dataset_name)
        for idx, row in dataset_eval_df.iterrows():
            model_eval_list_dict[row.model].append(float(row.testauc))

    row_list = []
    for model_name, auc_list in model_eval_list_dict.items():
        auc_list = [float(f"{x:.4f}") for x in auc_list]
        row_list.append([model_name] + auc_list)
    
    int_eval_df = pd.DataFrame(row_list, columns=["model name"] + dataset_name_list)
    int_eval_df.to_csv(os.path.join(eval_save_dir, f"pykt_eval_results.csv"), index=False) 

    logger.info(f"completed. {time.time() - st}")
