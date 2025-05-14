
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

kupypd_train_hp_search_folder = "search_hp"
kupypd_train_folder = "train_eval"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='0,1')
    parser.add_argument("--emb_model", default="unixcoder")
    parser.add_argument('--data_root_dir', default='kupypd_data')
    parser.add_argument("--all_accepted_reattent", action="store_true")
    parser.add_argument('--data_aux_dir', default="../..")
    parser.add_argument('--folds', type=int, default=None)
    parser.add_argument('--dataset_hp', default=None)
    parser.add_argument('--semester_groups', default=None)
    parser.add_argument('--hp_search', action="store_true")
    parser.add_argument('--solving_interactions', 
                        choices=[x.name for x in kupypd_config.solving_interaction_types], 
                        default=kupypd_config.solving_interaction_types.none.name)
    parser.add_argument('--result_dir', default="kupypd_result")
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()

    program_st = time.time()

    solving_interactions = kupypd_config.solving_interaction_types[args.solving_interactions]
    data_aux_path = os.path.join(args.data_aux_dir, "attempt_solving_interaction_aux.csv")
    if not os.path.isfile(data_aux_path):
        data_aux_path = None
    if data_aux_path is None:
        solving_interactions = kupypd_config.solving_interaction_types.none

    def get_save_dir(root_dir):
        if args.hp_search:
            folder = kupypd_train_hp_search_folder
        else:
            folder = kupypd_train_folder

        folder += "_si_" + solving_interactions.name
        save_dir = os.path.join(root_dir, folder)

        return save_dir + "_" + args.emb_model

    def get_postfix():
        postfix = ""
        if args.all_accepted_reattent:
            postfix += "_all_accepted_reattent"

        return postfix

    train_save_dir = get_save_dir(args.result_dir) + get_postfix()

    if not os.path.isdir(train_save_dir):
        os.makedirs(train_save_dir)

    logger_dir = get_save_dir("kupypd_log")

    logger_name = f"kupypd_train_emb_{args.emb_model}-devices_{args.devices}" + get_postfix()

    if args.debug:
        logger_name = "debug_" + logger_name

    import datetime
    logger_name += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    logger = make_logger.make(logger_name, time_filename=False, save_dir=logger_dir)

    arg_params = vars(args)
    logger.info(f"args:\n{arg_params}")

    gpu_devices = args.devices.split(',')

    data_root_dir = os.path.join(args.data_root_dir, f"{args.emb_model}")
    if args.all_accepted_reattent:
        data_root_dir += "_all_accepted_reattent"

    semester_group_dir_list = []
    for semester_group in sorted(os.listdir(data_root_dir)):
        dir_path = os.path.join(data_root_dir, semester_group)
        if not os.path.isdir(dir_path):
            continue
        semester_group_dir_list.append([semester_group, os.path.join(data_root_dir, semester_group)])

    if args.semester_groups is not None:
        new_data_dir_list = []
        semester_groups = set(args.semester_groups.split(","))
        for semester_group, group_dir in semester_group_dir_list:
            if semester_group in semester_groups:
                new_data_dir_list.append([semester_group, group_dir])
        semester_group_dir_list = new_data_dir_list

    default_config = kupypd_config.Config()

    default_config.solving_interactions = solving_interactions
    if solving_interactions != kupypd_config.solving_interaction_types.none:
        with open(os.path.join(args.data_aux_dir, "attempt_solving_interaction_vocab_size.json"), "r") as f:
            attempt_solving_interaction_vocab_size = json.load(f)
            default_config.solving_num_embeddings = attempt_solving_interaction_vocab_size[solving_interactions.name]

    hp_comb_list_or_dict = None
    if args.hp_search:
        dataset_hp_path = None  # search for all

        hp_scopes = {
            "lr": [0.005, 0.001, 0.0005],
        }

        hp_comb_list_or_dict = make_hp_combs(hp_scopes)

        logger.info("starting to search best hyper parameters")
    else:
        dataset_hp_fname = f"semester_group_best_hp_{args.emb_model}" + get_postfix()
        dataset_hp_path = dataset_hp_fname + ".json"

        if solving_interactions != kupypd_config.solving_interaction_types.none: # train with solving interactions, not search hp
            hp_comb_list = [{
                        "solving_interaction_len": 0,
                        "emb_solving_interaction_size": 0,
                }]

            solving_interaction_lens = [3,7]
            if solving_interactions == kupypd_config.solving_interaction_types.result_status:
                solving_interaction_lens = [x+1 for x in solving_interaction_lens]   # submission의 status까지 포함되었음

            hp_solving_interaction_comb_list = make_hp_combs({
                    "solving_interaction_len": solving_interaction_lens,
                    "emb_solving_interaction_size": [1, 2, 3, 5],
                })
            
            hp_comb_list_or_dict = hp_comb_list + hp_solving_interaction_comb_list

    if hp_comb_list_or_dict is not None:
        n_total_combs = sum([len(x) for x in hp_comb_list_or_dict.values()]) if isinstance(hp_comb_list_or_dict, dict) else len(hp_comb_list_or_dict)
        logger.info(f"starting: total combinations:{n_total_combs} for all semester groups:\n" + str(hp_comb_list_or_dict))

    if args.dataset_hp is not None:
        dataset_hp_path = args.dataset_hp
        
    train_info_list = make_train_list(default_config, dataset_hp_path, hp_comb_list_or_dict, semester_group_dir_list, train_save_dir)

    folds = args.folds
    if folds is None:
        folds = 100 if args.hp_search else 10

    if args.debug:
        train_info_list = train_info_list[:1]
        folds = 2
        # default_config.epochs = 2
        logger.info("started with debug mode")

    logger.info(f"folds:{folds}")

    from kupypd_train_list import train_list
    train_list(gpu_devices, args.hp_search, data_root_dir, train_info_list, data_aux_path, folds, logger, logger_name, logger_dir, args.debug)
    
    logger.info(f"end. elapse:{time.time()-program_st}")
