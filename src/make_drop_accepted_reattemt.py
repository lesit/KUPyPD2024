import pandas as pd
import json
import numpy as np
import argparse
import os
import shutil
from tqdm import tqdm

try:
    from dataset_config import *
    from utils import *
    import make_logger as make_logger
except Exception as e:
    from src.dataset_config import *
    from src.utils import *
    import src.make_logger as make_logger

def drop(main_df:pd.DataFrame, save_re_attempt_status_path = None, logger=make_logger.get_stdout_logger()):
    semester_group_list = sorted(list(main_df["semester_group"].unique()))

    semester_group_df_list = []
    drop_info_list = []
    for semester_group in semester_group_list:
        semester_group_main_df = main_df[main_df["semester_group"] == semester_group]
        groupby_problem = semester_group_main_df.groupby(['student_id', 'problem_id'])

        drop_list = []
        after_results = []

        for (student_id, problem_id), index in tqdm(groupby_problem.groups.items()):
            user_problem_interaction_df = semester_group_main_df.loc[index]
            user_problem_interaction_df = user_problem_interaction_df.sort_values(by=['timestamp'])
            final_accepted = False
            for _, interaction in user_problem_interaction_df.iterrows():
                accepted = interaction.accept_result == "success"
                if final_accepted:
                    drop_list.append(interaction.code_id)
                    after_results.append(1 if accepted else 0)
                else:
                    if accepted:
                        final_accepted = True

        n_org = len(semester_group_main_df)
        semester_group_main_df = semester_group_main_df[~semester_group_main_df.code_id.isin(drop_list)]
        semester_group_df_list.append(semester_group_main_df)

        n_new = len(semester_group_main_df)
        dropped = n_org - n_new

        mean_after_results = float(np.mean(after_results))
        drop_info = (n_org, n_new, dropped, dropped/n_org, mean_after_results, after_results)
        drop_info_list.append([semester_group, drop_info])
        logger.info(f"{semester_group}. drop_after_accepted. semester_group_main_df. {n_org} -> {n_new}. dropped: {dropped}. mean_after_results:{mean_after_results}")

    refined_main_df = pd.concat(semester_group_df_list, ignore_index=True)

    n_total = 0
    n_dropped = 0
    total_after_results = []
    for semester_group, (n_org, n_new, dropped, drop_ratio, mean_after_results, after_results) in drop_info_list:
        n_total += n_org
        n_dropped += dropped
        total_after_results.extend(after_results)

    dropped_dict = {k:(n_org, n_new, dropped, drop_ratio, mean_after_results) for k, (n_org, n_new, dropped, drop_ratio, mean_after_results, after_results) in drop_info_list}
    dropped_dict = {k:dropped_dict[k] for k in sorted(dropped_dict.keys())}

    total_mean_after_results = float(np.mean(total_after_results))
    logger.info(f"dropped: {n_total}. {n_dropped}. ratio: {n_dropped/n_total:.4f}. total_mean_after_results: {total_mean_after_results:.4f}" + json.dumps(dropped_dict, indent=3))

    if save_re_attempt_status_path is not None:
        from collections import defaultdict
        row_dict = defaultdict(list)
        colums = []
        for semester_group, (n_org, n_new, dropped, drop_ratio, mean_after_results) in dropped_dict.items():
            colums.append(semester_group)
            new_info_list = [str(n_org), str(dropped), f"{drop_ratio*100:.2f}%", f"{mean_after_results*100:.2f}%"]
            for idx, value in enumerate(new_info_list):
                row_dict[idx].append(value)
        
        row_list = [row_dict[idx] for idx in sorted(row_dict.keys())]
        df = pd.DataFrame(row_list, columns=colums)
        df.insert(0, "", ["total submissions", "re-attempts", "re-attempts ratio", "next attempt acceptance ratio"])
        df.to_csv(save_re_attempt_status_path, index=False)

    return refined_main_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_dir", default="../../../dataset")
    parser.add_argument("--result_dir", default="kupypd_data")
    parser.add_argument("--save_re_attempt_status", default="re_attempt_status.csv")
    
    args = parser.parse_args()

    dataset_root_dir = args.dataset_root_dir

    save_dir = args.result_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    submission_path = get_dataset_path(dataset_root_dir, DatasetType.submission_interaction)
    main_df = pd.read_csv(submission_path)

    refined_main_df = drop(main_df, args.save_re_attempt_status)
    refined_main_df.to_csv(os.path.join(save_dir, DatasetType.submission_interaction.name), index=False)
