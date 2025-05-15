import pandas as pd
import json
import numpy as np
import argparse
import os
import shutil
import itertools

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from kupypd_config import *
from kupypd_path_extract import path_extract
from kupypd_create_code_ast_path import *

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

    return problems_d, train_val_s, test_s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_dir", default="../../../dataset")
    parser.add_argument("--inc_accepted_reattempt", action="store_true")
    parser.add_argument("--result_dir", default="kupypd_data")
    parser.add_argument("--semester_group", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    import datetime
    logger_name = f"preprocess"
    if args.inc_accepted_reattempt:
        logger_name += "_inc_accepted_reattempt"
    logger_name += "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    if args.debug:
        logger_name = "debug_" + logger_name

    log_dir = os.path.join("kupypd_log", "preprocess")
    logger = make_logger.make(logger_name, time_filename=False, save_dir=log_dir)

    dataset_root_dir = args.dataset_root_dir
        
    st = time.time()
    logger.info("start")
    
    config = Config()

    save_root_dir = args.result_dir
    if args.inc_accepted_reattempt:
        save_root_dir += "_inc_accepted_reattempt"

    if os.path.isdir(save_root_dir):
        shutil.rmtree(save_root_dir)

    submission_path = get_dataset_path(dataset_root_dir, DatasetType.submission_interaction)
    main_df = pd.read_csv(submission_path)
    logger.info(f"main_df.shape:{main_df.shape}")
    if not args.inc_accepted_reattempt:
        import src.make_drop_accepted_reattemt as make_drop_accepted_reattemt
        main_df = make_drop_accepted_reattemt.drop(main_df, os.path.join("re_attempt_status.csv"), logger)
        logger.info(f"main_df.shape:{main_df.shape}")

    code_path = get_dataset_path(dataset_root_dir, DatasetType.submission_code)
    code_df = pd.read_csv(code_path)

    semester_group_list = sorted(list(main_df["semester_group"].unique()))

    if args.semester_group is not None:
        for semester_group in semester_group_list:
            if semester_group == args.semester_group:
                semester_group_list = [semester_group]
                break

    def group_preprocess(semester_group, shared_stat_dict):
        mp_logger_name = semester_group
        mp_log_dir = os.path.join(log_dir, logger_name, semester_group)
        proc_logger = make_logger.make(mp_logger_name, time_filename=False, save_dir=mp_log_dir)
        proc_logger.info(f"{semester_group}. group preprocess: {semester_group}. start")

        data_save_dir = os.path.join(save_root_dir, semester_group)
        if not os.path.isdir(data_save_dir):
            os.makedirs(data_save_dir)            
        
        gs = time.time()

        semester_group_main_df = main_df[main_df["semester_group"] == semester_group]
        semester_group_code_df = code_df[code_df["semester_group"] == semester_group]
        proc_logger.info(f"semester_group_main_df.shape:{semester_group_main_df.shape}")

        semester_group_main_df = semester_group_main_df.sort_values(by=['timestamp'])

        semester_group_main_df['score'] = np.array(semester_group_main_df["accept_result"] == "success").astype(int)  

        proc_logger.info(f"attempt_preprocess.start.")
        st_attempt = time.time()
        problems_d, train_val_s, test_s = attempt_preprocess(semester_group_main_df, data_save_dir, proc_logger)    
        proc_logger.info(f"attempt_preprocess.end. elapse: {time.time() - st_attempt}")

        st_extract = time.time()
        proc_logger.info(f"path_extract.start.")
        raw_paths_list, using_n_path_list, code_path_lengths_list = path_extract(semester_group_code_df,
                                                                                config,
                                                                                proc_logger, mp_logger_name, mp_log_dir)
        proc_logger.info(f"path_extract.end. elapse: {time.time() - st_extract}")

        # student_id는 CodeAstPath에서 사용된다.
        labeled_paths_df = pd.DataFrame(raw_paths_list, columns=["student_id", "CodeStateID", "RawASTPath"])
        labeled_paths_df.to_csv(os.path.join(data_save_dir, "labeled_paths.tsv"), sep="\t", header=True)
        
        st_code_node_path = time.time()
        CodeAstPath.save(data_save_dir, proc_logger)
        proc_logger.info(f"save_code_node_path.end. elapse: {time.time() - st_code_node_path}")

        shared_stat_dict[semester_group] = (using_n_path_list, code_path_lengths_list)
        proc_logger.info(f"{semester_group}. group preprocess: {semester_group}. end. elapse: {time.time() - gs}")

    if args.debug or len(semester_group_list)<=1:
        shared_stat_dict = dict()
        for semester_group in semester_group_list:
            group_preprocess(semester_group, shared_stat_dict)
    else: 
        import multiprocessing
        manager = multiprocessing.Manager()
        shared_stat_dict = manager.dict()

        process_list = []
        for semester_group in semester_group_list:
            process = multiprocessing.Process(target=group_preprocess, args=[semester_group, shared_stat_dict])
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

    import pickle
    for semester_group in sorted(shared_stat_dict.keys()):
        save_stat_info_dir = os.path.join("stats", "prep_ast_path_len_stats", semester_group)
        import shutil
        if os.path.isdir(save_stat_info_dir):
            shutil.rmtree(save_stat_info_dir)
        os.makedirs(save_stat_info_dir)            

        try:
            using_code_n_path_list, code_path_lengths_list = shared_stat_dict[semester_group]
            def get_stats(data):
                len_mean = float(f"{np.mean(data):.2f}")
                len_std = float(f"{np.std(data):.2f}")
                len_min = int(np.min(data))
                len_max = int(np.max(data))
                return len_mean, len_std, len_min, len_max

            def save_info(data, name):
                with open(os.path.join(save_stat_info_dir, f"{semester_group}_{name}.json"), "w") as f:
                    json.dump(data, f, indent=3)
            def save_data(data, name):
                with open(os.path.join(save_stat_info_dir, f"{semester_group}_{name}.pkl"), 'wb') as f:
                    pickle.dump(data, f)

            def save_list_stats(data, name):
                len_mean, len_std, len_min, len_max = get_stats(data)
                logger.info(f"{semester_group}. {name}. mean:{len_mean}, std:{len_std}, min:{len_min}, max:{len_max}")

                path_len_stats = {
                    "num": len(data),
                    "mean": len_mean,
                    "std": len_std,
                    "min": len_min,
                    "max": len_max,
                }
                save_info(path_len_stats, name)
                save_data(data, name)
                
            save_list_stats(using_code_n_path_list, "using_code_n_path_list")
            
            # code_path_lengths_list = [code_path_lengths for code_path_lengths in code_path_lengths_list if len(code_path_lengths)>0]
            
            org_code_n_path_list = [len(code_path_lengths) for code_path_lengths in code_path_lengths_list]
            save_list_stats(org_code_n_path_list, "org_code_n_path_list")

            for max_len in range(config.code_path_length+1, config.code_path_length+4):
                max_len_code_n_path_list = [len([path_len for path_len in code_path_lengths if path_len<=max_len]) for code_path_lengths in code_path_lengths_list]
                save_list_stats(max_len_code_n_path_list, f"max_len_{max_len}_code_n_path_list")

            logger.info(f"{semester_group}. get overall")
            overall_path_len_list = list(itertools.chain.from_iterable(code_path_lengths_list))

            logger.info(f"{semester_group}. save overall")
            save_list_stats(overall_path_len_list, "overall_path_len_list")
            
            logger.info(f"{semester_group}. save code_path_lengths_list")
            save_data(code_path_lengths_list, "code_path_lengths_list")

            logger.info(f"{semester_group}. get_stats of code_path_lengths_list")
            code_path_lengths_stats_list = [get_stats(code_path_lengths) for code_path_lengths in code_path_lengths_list if len(code_path_lengths)>0]
            def get_stats_value_list(stats_list):
                info = dict()            
                for idx, key in enumerate(["mean", "std", "min", "max"]):
                    info[key] = [x[idx] for x in stats_list]
                return info
            logger.info(f"{semester_group}. save stats of code_path_lengths_list")
            save_info(get_stats_value_list(code_path_lengths_stats_list), "code_path_lengths_stats_list")
            save_data(code_path_lengths_stats_list, "code_path_lengths_stats_list")

        except Exception as e:
            make_logger.write_exception_log(logger, e, f"at {semester_group}")

    logger.info(f"end.{time.time() - st}")
