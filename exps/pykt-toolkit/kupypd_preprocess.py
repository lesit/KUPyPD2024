import os, sys
import argparse
import os
import pandas as pd
import numpy as np

import sys
module_path = os.path.abspath("../../")
if module_path not in sys.path:
	sys.path.append(module_path)

import src.make_logger as make_logger
from src.dataset_config import *

from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question

def write_txt(file, data):
    with open(file, "w") as f:
        for dd in data:
            for d in dd:
                f.write(",".join(d) + "\n")

from datetime import datetime
def change2timestamp(t, hasf=True):
    if hasf:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    else:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    return int(timeStamp)

def format_list2str(input_list):
    return [str(x) for x in input_list]

def load_kupypd2024(submissions_df:pd.DataFrame, logger):
    """"
    Returns:
        dataframe: the merge df
    """
    logger.info(f"load_kupypd2024.start")

    logger.info(f"submissions_df: {submissions_df.shape}")
    submissions_df = submissions_df.sort_values(by=['timestamp'])

    submissions_df['answer_timestamp'] = submissions_df['timestamp'].apply(change2timestamp)
    submissions_df["is_correct"] = (submissions_df["accept_result"]=="success").astype(int)
   
    # problem_id에 간혹 _ 가 있어서, split_datasets_que.main 함수를 통해 split 되는 문제가 있음!!!!!!!!!!!!!
    # 그래서 _를 +로 변경함
    submissions_df['problem_id'] = submissions_df['problem_id'].str.replace('_', '+')
    # student_id는 학기 별로 구분짓기 위해 prefix를 넣으며 _를 사용했다. 따라서, 이것도 +로 변경함
    submissions_df['student_id'] = submissions_df['student_id'].str.replace('_', '+')

    logger.info(f"Num of student {submissions_df['student_id'].unique().size}")
    logger.info(f"Num of question {submissions_df['problem_id'].unique().size}")

    logger.info(f"load_kupypd2024.end")
    return submissions_df

def get_user_inters(df):
    """convert df to user sequences 

    Args:
        df (_type_): the merged df

    Returns:
        List: user_inters
    """
    user_inters = []
    for user, group in df.groupby("student_id", sort=False):
        group = group.sort_values(["answer_timestamp"], ascending=True)

        seq_problems = ["NA"]
        seq_skills = group['problem_id'].tolist()

        seq_ans = group['is_correct'].tolist()
        seq_start_time = group['answer_timestamp'].tolist()
        seq_response_cost = ["NA"]
        seq_len = len(group)
        user_inters.append(
            [[str(user), str(seq_len)],
             format_list2str(seq_problems),
             format_list2str(seq_skills),
             format_list2str(seq_ans),
             format_list2str(seq_start_time),
             format_list2str(seq_response_cost)])
    return user_inters


def read_data_from_csv(submissions_df, writef, logger):
    df = load_kupypd2024(submissions_df, logger)
    
    user_inters = get_user_inters(df)
    write_txt(writef, user_inters)

    return user_inters
    
configf = "configs/data_config.json"

from kupypd_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../../../dataset")
    parser.add_argument("--save_dir", type=str, default="kupypd_data")
    parser.add_argument("--log_dir", type=str, default="kupypd_log")
    parser.add_argument("-m","--min_seq_len", type=int, default=8)  # at least 8 week. 1 more problem per week.
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    parser.add_argument("--inc_accepted_reattempt", action="store_true")

    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    print(args)
    dataset_root_dir = args.dataset_dir

    save_data_root_dir = args.save_dir
    if args.inc_accepted_reattempt:
        save_data_root_dir += "_inc_accepted_reattempt"

    logger = make_logger.make("kupypd_preprocess", save_dir=os.path.join(args.log_dir, "preprocess"))

    logger.info(f"start")

    if os.path.isdir(save_data_root_dir):
        import shutil
        shutil.rmtree(save_data_root_dir)

    total_user_inters = []

    submissions_path = get_dataset_path(dataset_root_dir, DatasetType.submission_interaction)
    total_submissions_df = pd.read_csv(submissions_path)
    logger.info(f"total_submissions_df: {total_submissions_df.shape}")
    if not args.inc_accepted_reattempt:
        drop_accepted_reattempt_dir = "drop_accepted_reattempt"
        import src.make_drop_accepted_reattemt as make_drop_accepted_reattemt
        total_submissions_df = make_drop_accepted_reattemt.drop(total_submissions_df, os.path.join("re_attempt_status.csv"), logger)
    
        logger.info(f"total_submissions_df: {total_submissions_df.shape}")

    semester_group_list = sorted(list(total_submissions_df["semester_group"].unique()))
    for semester_group in semester_group_list:
        logger.info(f"load. {semester_group}")
        submissions_df = total_submissions_df[total_submissions_df["semester_group"] == semester_group]

        data_save_dir = os.path.join(save_data_root_dir, semester_group)
        if not os.path.isdir(data_save_dir):
            os.makedirs(data_save_dir)            
        writef = os.path.join(data_save_dir, "data.txt")

        user_inters = read_data_from_csv(submissions_df, writef, logger)

        print("-"*50)
        print(f"dname: {data_save_dir}, writef: {writef}")
        # split
        os.system("rm " + data_save_dir + "/*.pkl")

        dataset_name = semester_group
        if args.inc_accepted_reattempt:
            dataset_name += "_inc_accepted_reattempt"

        #for concept level model
        split_concept(data_save_dir, writef, dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
        print("="*100)

        #for question level model
        split_question(data_save_dir, writef, dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)

        total_user_inters.extend(user_inters)
        logger.info("")

    logger.info(f"completed")
