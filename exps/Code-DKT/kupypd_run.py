
import os
import time
import json

import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")

import kupypd_config

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger
from src.utils import *

from kupypd_train_info import make_train_list

kupypd_train_hp_search_folder = "train_hp_search"
kupypd_train_folder = "train"
kupypd_train_ablation_folder = "train_ablation_study"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--data_root_dir', default='kupypd_data')
    parser.add_argument("--inc_accepted_reattempt", action="store_true")
    parser.add_argument('--data_aux_root_dir', default="../../kupypd_data_aux")
    parser.add_argument('--folds', type=int, default=None)
    parser.add_argument('--dataset_hp', default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--semester_groups', default=None)
    parser.add_argument('--hp_search', action="store_true")
    parser.add_argument('--ablation_study', action="store_true")
    parser.add_argument('--result_dir', default="kupypd_result")
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()

    program_st = time.time()

    def get_save_dir(root_dir):
        if args.hp_search:
            folder = kupypd_train_hp_search_folder
        elif args.ablation_study:
            folder = kupypd_train_ablation_folder
        else:
            folder = kupypd_train_folder

        save_dir = os.path.join(root_dir, folder)

        return save_dir

    train_save_dir = get_save_dir(args.result_dir)
    if args.inc_accepted_reattempt:
        train_save_dir += "_inc_accepted_reattempt"

    if not os.path.isdir(train_save_dir):
        os.makedirs(train_save_dir)

    logger_dir = get_save_dir("kupypd_log")

    logger_name = f"kupypd_train-device_{args.devices}"
    if args.inc_accepted_reattempt:
        logger_name += "_inc_accepted_reattempt"

    if args.debug:
        logger_name = "debug_" + logger_name

    import datetime
    logger_name += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    logger = make_logger.make(logger_name, time_filename=False, save_dir=logger_dir)

    arg_params = vars(args)
    logger.info(f"args:\n{arg_params}")

    gpu_devices = args.devices.split(',')

    data_root_dir = args.data_root_dir
    if args.inc_accepted_reattempt:
        data_root_dir += "_inc_accepted_reattempt"

    semester_group_dir_list = []
    for semester_group in sorted(os.listdir(data_root_dir)):
        semester_group_dir_list.append([semester_group, os.path.join(data_root_dir, semester_group)])

    if args.semester_groups is not None:
        new_data_dir_list = []
        semester_groups = set(args.semester_groups.split(","))
        for semester_group, group_dir in semester_group_dir_list:
            if semester_group in semester_groups:
                new_data_dir_list.append([semester_group, group_dir])
        semester_group_dir_list = new_data_dir_list

    default_config = kupypd_config.Config()

    hp_comb_list_or_dict = None
    if args.hp_search:
        dataset_hp_path = None  # search for all

        hp_scopes = {
            "emb_size": [50, 100, 150],
            "lr": [0.0005, 0.001, 0.005],
        }

        hp_comb_list_or_dict = make_hp_combs(hp_scopes)

        for hp_comb in hp_comb_list_or_dict:
            if hp_comb["emb_size"] == 300:
                hp_comb["batch_size"] = 16

        logger.info("starting to search best hyper parameters")
    else:
        dataset_hp_fname = "semester_group_best_hp"
        if args.inc_accepted_reattempt:
            dataset_hp_fname += "_inc_accepted_reattempt"
        dataset_hp_path = dataset_hp_fname + ".json"

        if args.ablation_study:
            bottom_freq_cumsum_1p_last_freq_list = [10, 5, 16, 13, 6]
            bottom_freq_cumsum_5p_last_freq_list = [93, 31, 219, 192, 51]
            bottom_freq_cumsum_10p_last_freq_list = [272, 79, 733, 717, 170]

            bottom_freq_cumsum_last_freq_list = zip([0,0,0,0,0],
                                                    bottom_freq_cumsum_1p_last_freq_list,
                                                    bottom_freq_cumsum_5p_last_freq_list, 
                                                    bottom_freq_cumsum_10p_last_freq_list)
            hp_comb_list_or_dict = dict()
            for (semester_group, _), last_freqs in zip(semester_group_dir_list, bottom_freq_cumsum_last_freq_list):
                hp_combs = make_hp_combs({
                    "abandon_low_frequence": last_freqs,
                })
                hp_comb_list_or_dict[semester_group] = hp_combs

    if args.batch_size is not None:
        for hp_comb in hp_comb_list_or_dict:
            hp_comb["batch_size"] = args.batch_size

    if hp_comb_list_or_dict is not None:
        n_total_combs = sum([len(x) for x in hp_comb_list_or_dict.values()]) if isinstance(hp_comb_list_or_dict, dict) else len(hp_comb_list_or_dict)
        logger.info(f"starting: total combinations:{n_total_combs} for all semester groups:\n" + str(hp_comb_list_or_dict))

    folds = args.folds
    if folds is None:
        folds = 100 if args.hp_search else 10

    if args.debug:
        folds = 2
        # default_config.epochs = 2
        logger.info("started with debug mode")

    logger.info(f"folds:{folds}")

    if args.dataset_hp is not None:
        dataset_hp_path = args.dataset_hp
        
    data_aux_root_dir = args.data_aux_root_dir

    if not os.path.isdir(data_aux_root_dir):
        data_aux_root_dir = None

    train_info_list = make_train_list(default_config, dataset_hp_path, hp_comb_list_or_dict, semester_group_dir_list, data_aux_root_dir, train_save_dir)

    from kupypd_train_list import train_list
    train_list(gpu_devices, args.hp_search, train_info_list, folds, logger, logger_name, logger_dir, args.debug)
    
    logger.info(f"end. elapse:{time.time()-program_st}")
