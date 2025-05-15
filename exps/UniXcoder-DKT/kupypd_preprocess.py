import pandas as pd
import json
import numpy as np
import argparse
import os
import shutil
from distutils import dir_util

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from kupypd_config import *

import sys
import time

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger
from src.dataset_config import *
from src.utils import *

def attempt_preprocess(main_df, save_dir, logger):
    main_df = main_df.sort_values(by=['timestamp'])
    students = pd.unique(main_df["student_id"])
    problems = pd.unique(main_df["problem_id"])
    logger.info(f"students.shape:{students.shape}")
    logger.info(f"problems.shape:{problems.shape}")

    problems_d = {k:v for (v,k) in enumerate(problems) }

    d = {}
    for s in students:
        d[s] = {}
        df = main_df[main_df["student_id"] == s]
        d[s]["length"] = len(df)    # 해당 학생의 실습 개수(execution 또는 submission)
        d[s]["Problems"] = [str(problems_d[i]) for i in df["problem_id"]]
        results = (df["score"]==1).astype(int)
        d[s]["Result"] = list(results.astype(str))
        d[s]["mean"] = results.mean()
        d[s]["CodeStates"] = [str(x) for x in df["code_id"]]

    train_val_s, test_s = train_test_split(students, test_size=0.2, random_state=1)

    pypd_dkt_features_dir = os.path.join(save_dir, "DKTFeatures")

    if not os.path.isdir(pypd_dkt_features_dir):
        os.makedirs(pypd_dkt_features_dir)

    np.save(os.path.join(save_dir, "training_students.npy"), train_val_s)
    np.save(os.path.join(save_dir, "testing_students.npy"), test_s)

    np.save(os.path.join(save_dir, "problems.npy"), problems_d)

    def save_feature_data(data_students, filename):
        values = []
        for s in data_students:
            if d[s]['length']>0:
                length = str(d[s]['length'])
                CodeStates = ",".join(d[s]['CodeStates'])
                Problems = ",".join(d[s]['Problems'])
                Result = ",".join(d[s]['Result'])
                mean = d[s]["mean"]
                values.append([s, length, CodeStates, Problems, Result, mean])

        df = pd.DataFrame(data=values, columns=["student", "length", "CodeStates", "Problems", "Result", "Result_mean"])
        df.to_csv(os.path.join(pypd_dkt_features_dir, filename), index=False, encoding="utf-8-sig")

        mean = df.Result_mean.mean()
        logger.info(f"{filename}. result mean:{mean}")

    save_feature_data(test_s, "test_data.csv")
    save_feature_data(train_val_s, "train_data.csv")

    # jae: created 100 samples using the same data by only varying the random state.
    for fold in range(100):
        train_s, val_s = train_test_split(train_val_s, test_size=0.25, random_state=fold)

        # 아래 두개는 hyper-parameter searching에만 사용해야 함!!!
        np.save(os.path.join(pypd_dkt_features_dir, f"training_train_students_{fold}.npy"), train_s)
        np.save(os.path.join(pypd_dkt_features_dir, f"training_val_students_{fold}.npy"), val_s)

    return train_val_s, test_s

from kupypd_code_emb import save_code_embed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='0,1')
    parser.add_argument("--dataset_root_dir", default="../../../dataset")
    parser.add_argument("--inc_accepted_reattempt", action="store_true")
    parser.add_argument("--emb_model", default="unixcoder")
    parser.add_argument("--result_dir", default="kupypd_data")
    parser.add_argument("--semester_group", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    import datetime
    logger_name = f"preprocess_emb_{args.emb_model}-devices_{args.devices}"
    if args.inc_accepted_reattempt:
        logger_name += "_inc_accepted_reattempt"
    logger_name += "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    if args.debug:
        logger_name = "debug_" + logger_name

    gpu_devices = args.devices.split(',')

    log_dir = os.path.join("kupypd_log", f"preprocess_emb_{args.emb_model}")
    logger = make_logger.make(logger_name, time_filename=False, save_dir=log_dir)

    dataset_root_dir = args.dataset_root_dir
        
    st = time.time()
    logger.info("start")
    
    config = Config()

    save_root_dir = os.path.join(args.result_dir, f"{args.emb_model}")
    if args.inc_accepted_reattempt:
        save_root_dir += "_inc_accepted_reattempt"

    if os.path.isdir(save_root_dir):
        shutil.rmtree(save_root_dir)
    os.makedirs(save_root_dir)

    st_save_embed = time.time()
    logger.info(f"save_code_embed.start.")
    code_path = get_dataset_path(dataset_root_dir, DatasetType.submission_code)
    code_df = pd.read_csv(code_path)
    logger.info(f"code_df.shape:{code_df.shape}")
    save_code_embed(gpu_devices, args.emb_model, code_df, save_root_dir, logger, logger_name, log_dir)
    logger.info(f"save_code_embed.end. elapse: {time.time() - st_save_embed}")

    submission_path = get_dataset_path(dataset_root_dir, DatasetType.submission_interaction)
    main_df = pd.read_csv(submission_path)
    logger.info(f"main_df.shape:{main_df.shape}")
    if not args.inc_accepted_reattempt:
        import src.make_drop_accepted_reattemt as make_drop_accepted_reattemt
        main_df = make_drop_accepted_reattemt.drop(main_df, os.path.join(save_root_dir, "re_attempt_status.csv"), logger)
        logger.info(f"main_df.shape:{main_df.shape}")

    semester_group_list = sorted(list(main_df["semester_group"].unique()))

    if args.semester_group is not None:
        for semester_group in semester_group_list:
            if semester_group == args.semester_group:
                semester_group_list = [semester_group]
                break
    if args.debug:
        semester_group_list = semester_group_list[:1]
    def group_preprocess(semester_group, proc_logger):
        proc_logger.info(f"group preprocess: {semester_group}. start")

        data_save_dir = os.path.join(save_root_dir, semester_group)
        if not os.path.isdir(data_save_dir):
            os.makedirs(data_save_dir)            
        
        gs = time.time()

        semester_group_main_df = main_df[main_df["semester_group"] == semester_group]
        proc_logger.info(f"semester_group_main_df.shape:{semester_group_main_df.shape}")

        semester_group_main_df = semester_group_main_df.sort_values(by=['timestamp'])
        semester_group_main_df['score'] = np.array(semester_group_main_df["accept_result"] == "success").astype(int)  

        proc_logger.info(f"attempt_preprocess.start.")
        st_attempt = time.time()
        train_val_s, test_s = attempt_preprocess(semester_group_main_df, data_save_dir, proc_logger)    
        proc_logger.info(f"attempt_preprocess.end. elapse: {time.time() - st_attempt}")

        proc_logger.info(f"group preprocess: {semester_group}. end. elapse: {time.time() - gs}")

    if args.debug:
        for semester_group in semester_group_list:
            group_preprocess(semester_group, logger)
    else: 
        import multiprocessing

        manager = multiprocessing.Manager()

        def mp_preprocess(logger_name, semester_group, debug):
            mp_logger = make_logger.make(f"{logger_name}_{semester_group}", time_filename=False, save_dir=log_dir)

            mp_st = time.time()
            mp_logger.info(f"train models with multiprocessing: {semester_group}. start")

            try:
                group_preprocess(semester_group, mp_logger)
            except Exception as e:
                make_logger.write_exception_log(mp_logger, e)
                exit(-1)

            mp_logger.info("")

            mp_logger.info(f"train models with multiprocessing: {semester_group}. end. elapse:{time.time()-mp_st}")

        process_list = []
        for semester_group in semester_group_list:
            process = multiprocessing.Process(target=mp_preprocess, args=[logger_name, semester_group, args.debug])
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

    logger.info(f"end.{time.time() - st}")
