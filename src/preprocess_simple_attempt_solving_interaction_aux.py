
import os
import numpy as np
import argparse
import pandas as pd
import json
import sys
import time

# module_path = os.path.abspath("../..")
# if module_path not in sys.path:
#     sys.path.append(module_path)

import make_logger
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_dir", default="../../dataset")
    parser.add_argument("--result_dir", default="../")
    args = parser.parse_args()

    from datetime import datetime
    log_dir = os.path.join("kupypd_log", "preprocess_submission_aux", datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    logger_name = "preprocess_submission_aux"
    logger = make_logger.make(logger_name, time_filename=False, save_dir=log_dir)

    st = time.time()
    logger.info("start")

    dataset_root_dir = args.dataset_root_dir
    save_root_dir = args.result_dir

    status_indices = {"empty":"0"}

    execution_path = get_dataset_path(dataset_root_dir, DatasetType.execution_interaction)
    exe_df = pd.read_csv(execution_path)
    exe_df = exe_df.dropna(subset=["error_name"])
    exe_df.error_name = exe_df.error_name.fillna("")
    # status_indices = {None:0}
    error_names = set(exe_df["error_name"].unique())
    status_indices["no_error"] = "1"
    start = len(status_indices)
    status_indices.update({name:str(start+idx) for idx, name in enumerate(sorted(list(error_names)))})

    submission_path = get_dataset_path(dataset_root_dir, DatasetType.submission_interaction)
    sub_df = pd.read_csv(submission_path)

    start = len(status_indices)
    status_indices.update({"AC":str(start)})

    verdicts = set(sub_df["verdict"].unique())
    verdicts = verdicts.difference({"AC"})
    start = len(status_indices)
    status_indices.update({name:str(start+idx) for idx, name in enumerate(sorted(list(verdicts)))})

    status_indices["fb"] = str(len(status_indices))

    attempt_solving_interaction_vocab_size = {
        "simple": 4,
        "result_status": len(status_indices),
        "status_indices": status_indices
    }
    with open(os.path.join(save_root_dir, "attempt_solving_interaction_vocab_size.json"), "w") as f:
        json.dump(attempt_solving_interaction_vocab_size, f, indent=3)

    attempt_interaction_sequence_dfs = load_attempt_interaction_sequence(dataset_root_dir)
    submission_aux_list = []
    for (semester_group, student_id, problem_id), interactions_df_list  in attempt_interaction_sequence_dfs.items():
        for interactions_df in interactions_df_list:
            # interaction_columns: "type", "code_id", "result", "result_status", 'timestamp', 'n_error_feedback', 'error_feedback_ids'
            time_format = '%Y-%m-%d %H:%M:%S.%f'

            timestamp_list = []
            simple_encodes = []
            status_represented_encodes = []
            sum_success_executions = 0
            sum_failure_executions = 0
            sum_ai_error_feedback_usage = 0

            for idx, row in interactions_df.iterrows():
                timestamp = datetime.strptime(row.timestamp, time_format)
                timestamp_list.append(timestamp)

                if len(row.result_status) == 0:
                    status_idx = status_indices["no_error"]
                else:
                    status_idx = status_indices[row.result_status]
                status_represented_encodes += [status_idx]
                status_represented_encodes += [status_indices["fb"]] * row.n_error_feedback

                interaction_type = InteractionType[row.type]
                if interaction_type != InteractionType.submission:
                    if len(row.result_status) > 0:
                        sum_success_executions += 1
                        simple_encodes.append("1")
                    else:
                        sum_failure_executions += 1
                        simple_encodes.append("2")
                    if row.n_error_feedback>0:
                        simple_encodes += ["3"] * row.n_error_feedback

                    sum_ai_error_feedback_usage += row.n_error_feedback
                else:
                    if len(timestamp_list)>0:
                        duration = timestamp-timestamp_list[0]
                        duration = duration.total_seconds()
                    else:
                        duration = 0

                    submission_aux_list.append([semester_group,
                                                row.code_id,
                                                1 if row.result == "success" else 0,
                                                ",".join(simple_encodes),
                                                ",".join(status_represented_encodes),
                                                sum_success_executions+sum_failure_executions,
                                                sum_success_executions,
                                                sum_failure_executions,
                                                sum_ai_error_feedback_usage,
                                                duration
                                                ])

    aux_df = pd.DataFrame(submission_aux_list, columns=["semester_group", "submission_id", 
                                                        "submission_result",
                                                        "simple_encodes",
                                                        "status_represented_encodes",
                                                        "total_executions", 
                                                        "exe_success", "exe_failure", "exe_ai_err_feedbak_usage", 
                                                        "duration"])
    aux_df.to_csv(os.path.join(save_root_dir, f"attempt_solving_interaction_aux.csv"), index=False)

    logger.info(f"group preprocess: {semester_group}. end. elapse: {time.time() - st}")

        
    # st = time.time()
    # logger.info("start")
    
    # save_root_dir = args.result_dir
    # if not os.path.isdir(save_root_dir):
    #     os.makedirs(save_root_dir)            

    # for data_dir_info in data_dir_list:
    #     semester_group, group_dir = data_dir_info

    #     logger.info(f"group preprocess: {semester_group}. start")

    #     gs = time.time()

    #     code_executions_path = os.path.join(group_dir, "execution_interaction.csv")
    #     exe_df = pd.read_csv(code_executions_path)
    #     exe_df.error_name = exe_df.error_name.fillna("")
    #     exe_error_types = {name:idx for idx, name in enumerate(sorted(list(set(exe_df["error_name"].unique()))))}

    #     solving_interaction_histories_path = os.path.join(group_dir, "problem_attempt_sequence.csv")
    #     main_df = pd.read_csv(solving_interaction_histories_path)
    #     logger.info(f"problem_attempt_sequence.shape:{main_df.shape}")

    #     submission_aux_list = []
    #     for idx, row in main_df.iterrows():
    #         problem_attempt_sequence = row["problem_attempt_sequence"]
    #         problem_attempt_sequence = json.loads(problem_attempt_sequence)

    #         solving_interactions = ""
    #         sum_success_executions = 0
    #         sum_failure_executions = 0
    #         sum_ai_error_feedback_usage = 0
    #         timestamp_list = []

    #         for interaction in problem_attempt_sequence["list"]:
    #             interaction_type, id, result, result_status, timestamp, ai_error_feedback_usage = interaction

    #             time_format = '%Y-%m-%d %H:%M:%S.%f'
    #             timestamp = datetime.strptime(timestamp, time_format)
    #             if interaction_type != "submission":
    #                 if result == "success":
    #                     sum_success_executions += 1
    #                     solving_interactions += "1"
    #                 else:
    #                     sum_failure_executions += 1
    #                     solving_interactions += "2"
    #                 if ai_error_feedback_usage>0:
    #                     if ai_error_feedback_usage>1:
    #                         a = 0
    #                     solving_interactions += "3" * ai_error_feedback_usage

    #                 sum_ai_error_feedback_usage += ai_error_feedback_usage

    #                 timestamp_list.append(timestamp)
    #             else:
    #                 if len(timestamp_list)>0:
    #                     duration = timestamp-timestamp_list[0]
    #                     duration = duration.total_seconds()
    #                 else:
    #                     duration = 0

    #                 submission_aux_list.append([id,
    #                                             1 if result == "success" else 0,
    #                                             solving_interactions,
    #                                             len(solving_interactions),
    #                                             sum_success_executions+sum_failure_executions,
    #                                             sum_success_executions,
    #                                             sum_failure_executions,
    #                                             sum_ai_error_feedback_usage,
    #                                             duration
    #                                             ])

    #                 solving_interactions = ""
    #                 sum_success_executions = 0
    #                 sum_failure_executions = 0
    #                 sum_ai_error_feedback_usage = 0
    #                 timestamp_list = []

    #     aux_df = pd.DataFrame(submission_aux_list, columns=["submission_id", "submission_result",
    #                                                         "solving_interactions", "iterations_cnt",
    #                                                         "total_executions", "exe_success", "exe_failure", "exe_ai_err_feedbak_usage", "duration"])
    #     mean_row = aux_df.mean(numeric_only=True).round(2)
    #     mean_row = mean_row.to_frame().transpose()
    #     mean_row.insert(0, "submission_id", "mean")
    #     mean_row.insert(2, "solving_interactions", "")
    #     std_row = aux_df.std(numeric_only=True).round(2)
    #     std_row = std_row.to_frame().transpose()
    #     std_row.insert(0, "submission_id", "std")
    #     std_row.insert(2, "solving_interactions", "")
    #     aux_df = pd.concat([mean_row, std_row, aux_df], ignore_index=True)

    #     aux_df.to_csv(os.path.join(save_root_dir, f"{semester_group}_submission_aux.csv"), index=False)

    #     logger.info(f"group preprocess: {semester_group}. end. elapse: {time.time() - gs}")

    # logger.info("end")
