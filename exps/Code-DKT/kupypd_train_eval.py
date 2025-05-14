

import random
import numpy as np
import time

import kupypd_config
from kupypd_dataloader import get_data_loader
from kupypd_evaluation import *

from kupypd_create_code_ast_path import *
from c2vRNNModel import c2vRNNModel

def save_performance(is_early_stop:bool, result_df_list, save_path, hp_comb:dict=None):
    # ["fold", "epoch", "auc", "f1", "recall", "precision", "acc", "first_auc", "first_f1", "first_recall", "first_precision", "first_acc"]
    scores_df = pd.concat(result_df_list, ignore_index=True)
    scores_df = scores_df.sort_values(by=["fold"])
    if is_early_stop:
        mean_row = scores_df.mean(numeric_only=True).to_frame().transpose()
        mean_row["fold"] = "mean"
        scores_df = pd.concat([mean_row, scores_df], ignore_index=True)
    else:
        epoch_list = sorted(list(set(scores_df["epoch"].unique())))
        epoch_mean_row_dfs = []
        for epoch in epoch_list:
            epoch_score = scores_df[scores_df["epoch"] == epoch]
            mean_row = epoch_score.mean(numeric_only=True).to_frame().transpose()
            mean_row["fold"] = "epoch_mean"
            epoch_mean_row_dfs.append(mean_row)
        epoch_mean_row_dfs.append(scores_df)
        scores_df = pd.concat(epoch_mean_row_dfs, ignore_index=True)
    
    scores_df["epoch"] = scores_df["epoch"].round()
    scores_df = scores_df.astype({"epoch":"int32"})
    scores_df = scores_df.round(4)

    if hp_comb is not None:
        for k, v in reversed(hp_comb.items()):
            scores_df.insert(0, k, v)

    scores_df.to_csv(save_path, index=False)

from kupypd_cache_data import CacheDataLoad

import json
def train(is_hp_search, config:kupypd_config.Config, data_dir, cached_data:CacheDataLoad,
          folds, fold_result_dir, save_result_path, logger, log_prefix, es_paitence_diff=1e-3, es_paitence_epoch=20, **kwargs):
    import torch
    import torch.optim as optim
    def setup_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    setup_seed(0)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    run_st = time.time()

    if not os.path.isdir(fold_result_dir):
        os.makedirs(fold_result_dir)

    if os.path.isfile(save_result_path):
        logger.info(f"{log_prefix}. already result saved: {save_result_path}")
        return

    logger.info(f"{log_prefix}. result save: {save_result_path}")
    logger.info(f"config:"+json.dumps(config.__dict__, indent=3))

    cached_data.load_data(is_hp_search=is_hp_search, config=config, data_dir=data_dir)

    ast_node_count = cached_data.reader.ast_node_count
    ast_path_count = cached_data.reader.ast_path_count
    n_problems = cached_data.reader.numofques

    is_early_stop = is_hp_search   # when searching hp, stop early

    train_msg = f"{log_prefix}. problems: {n_problems}. bs:{config.batch_size}, emb:{config.emb_size}, lr:{config.lr}, es:{is_early_stop}"
    logger.info(f"{train_msg}. start")

    if is_early_stop:
        best_auc_list = []

    for fold in range(folds):
        fold_result_path = os.path.join(fold_result_dir, f"fold{fold}_result.csv")

        fold_log_prefix = f"{log_prefix}. fold:{fold}/{folds-1}"
        logger.info(f"{fold_log_prefix}. save:{fold_result_path} start")

        fold_st = time.time()
        logger.info(f"{fold_log_prefix} create data loader.start")

        if is_hp_search:
            train_s = np.load(os.path.join(cached_data.dkt_features_dir, f"training_train_students_{fold}.npy"),allow_pickle=True)
            val_s = np.load(os.path.join(cached_data.dkt_features_dir, f"training_val_students_{fold}.npy"),allow_pickle=True)

            fold_train_data = []
            for student in train_s:
                if kwargs.get("debug", False):
                    if student not in cached_data.train_data:
                        continue

                fold_train_data.append(cached_data.train_data[student])
            fold_train_data = np.asarray(fold_train_data)

            fold_test_data = []
            for student in val_s:
                if kwargs.get("debug", False):
                    if student not in cached_data.train_data:
                        continue

                fold_test_data.append(cached_data.train_data[student])
            fold_test_data = np.asarray(fold_test_data)
        else:
            fold_train_data = np.asarray(list(cached_data.train_data.values()))
            fold_test_data = cached_data.np_test_data

        train_loader, test_loader = get_data_loader(config.batch_size, fold_train_data, fold_test_data)
        logger.info(f"{fold_log_prefix} create data loader.end elapse:{time.time()-fold_st:.3f}")

        logger.info(f"{fold_log_prefix} train")
        model = c2vRNNModel(device,
                            n_problems * 2,
                            config.hidden,
                            config.layers,
                            n_problems,
                            emb_size=config.emb_size,
                            node_count=ast_node_count,
                            path_count=ast_path_count,
                            max_code_len=config.max_code_len) 

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        loss_func = lossFunc(n_problems, cached_data.reader.entire_maxstep, device)

        best_auc = 0
        best_epoch = -1

        epoch_results = []

        epoch_range = list(range(1, config.epochs+1))
        for epoch in epoch_range:
            model, optimizer, loss = train_epoch(model, train_loader, optimizer,
                                                              loss_func, n_problems, device)
            if is_early_stop:
                first_total_scores, first_scores, scores, performance = test_epoch(
                    model, test_loader, loss_func, device, n_problems, cached_data.reader.entire_maxstep)
                
                valid_auc = performance[0]

                if valid_auc > best_auc + es_paitence_diff:
                    best_auc = valid_auc
                    best_epoch = epoch

                    epoch_results = [[best_epoch] + performance + first_total_scores]

                print_log = epoch % 5 == 0 or epoch > epoch_range[-1]-5
                if print_log:
                    logger.info(f"{fold_log_prefix} epoch: {epoch}. valid auc: {valid_auc:.4f}. best auc: {best_auc} best epoch: {best_epoch}")

                if epoch - best_epoch >= es_paitence_epoch:
                    if not print_log:
                        logger.info(f"{fold_log_prefix} epoch: {epoch}. valid auc: {valid_auc:.4f}. best auc: {best_auc} best epoch: {best_epoch}")
                    break
            else:
                if epoch % 5 == 0 or epoch > epoch_range[-1]-5:
                    first_total_scores, first_scores, scores, performance = test_epoch(
                        model, test_loader, loss_func, device, n_problems, cached_data.reader.entire_maxstep)

                    epoch_results.append([epoch] + performance + first_total_scores)
                    eval_auc = performance[0]
                    logger.info(f"{fold_log_prefix} epoch: {epoch}. eval auc: {eval_auc:.4f}. train loss: {loss:.5f}")

        if is_early_stop:
            logger.info(f"result:\n"+"\n".join([str(x) for x in epoch_results]))
            best_auc_list.append(best_auc)
            logger.info(f"from 0 fold to now({fold} fold) , mean of auc: {np.mean(best_auc_list):.4f}")

        logger.info(f"{fold_log_prefix}. train end. epoch: {epoch}")

        score_columns = ["auc", "f1", "recall", "precision", "acc"]
        first_score_columns = [f"first_{x}" for x in score_columns]
        result_df = pd.DataFrame(epoch_results, columns=["epoch"] + score_columns + first_score_columns)
        result_df.insert(0, "fold", fold)
        result_df.to_csv(fold_result_path, index=False)

        logger.info(f"{fold_log_prefix}. end. elapse:{time.time()-fold_st:.3f}")
        logger.info("")
    
    result_df_list = []
    for fname in sorted(os.listdir(fold_result_dir)):
        if not fname.endswith("result.csv"):
            continue

        fold_result_path = os.path.join(fold_result_dir, fname)
        result_df = pd.read_csv(fold_result_path)
        
        result_df_list.append(result_df)

    save_performance(is_early_stop, result_df_list, save_result_path, kwargs.get("hp_comb"))

    torch.cuda.empty_cache()

    logger.info(f"{train_msg}. end. elapse:{time.time()-run_st}")
