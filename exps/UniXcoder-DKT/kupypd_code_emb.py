import os
import pandas as pd
import numpy as np
import pickle
from kupypd_config import Config
import sys

code_embed_fname = "code_embed.pkl"

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger

class CodeEmbed:
    def __init__(self, attempt_code_emb):
        self.attempt_code_emb = attempt_code_emb
        self.emb_size = len(list(attempt_code_emb.values())[0])

    def get_code_emb(self, id):
        return self.attempt_code_emb[id]

def save_code_embed(gpu_devices:list, emb_model_name, code_df:pd.DataFrame, embed_save_dir, logger, logger_name, log_dir):
    def get_embed_process(gpu, shared_result_dict, shared_lock):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        if emb_model_name == "unixcoder":
            from EmbedUniXcoder import get_code_embedding
        else:
            assert False

        mp_logger = make_logger.make(f"{logger_name}_gpu{gpu}", time_filename=False, save_dir=log_dir)

        mp_logger.info(f"get_embed_process: gpus:{gpu}. start")

        processed = 0
        other_processor_embed_list= []
        for idx, row in code_df.iterrows():
            code_id = row.code_id

            if shared_lock is not None:
                shared_lock.acquire()
                if code_id in shared_result_dict:
                    other_processor_embed_list.append(code_id)
                    shared_lock.release()
                    continue
            
                shared_result_dict[code_id] = None
                shared_lock.release()
            
            code = row.code
                
            try:
                code_embed, org_token_len, cur_token_len = get_code_embedding(code)
            except Exception as e:
                make_logger.write_exception_log(mp_logger, e)
                exit(-1)

            shared_result_dict[code_id] = code_embed, row.semester_group, org_token_len, cur_token_len

            processed += 1
            if processed % 1000 == 0:
                mp_logger.info(f"get_embed_process: gpus:{gpu}. get_embed_process: processed:{processed}")

        mp_logger.info(f"get_embed_process: gpus:{gpu}. processed:{processed}. other processor embed:{len(other_processor_embed_list)}\n:"+"\n".join(other_processor_embed_list))

        mp_logger.info(f"get_embed_process: gpus:{gpu}. end")

    logger.info("save_code_embed.start")
    if len(gpu_devices)<2:
        shared_result_dict = dict()
        get_embed_process(gpu_devices[0], shared_result_dict, shared_lock=None)
    else:
        import multiprocessing

        shared_lock = multiprocessing.Lock()

        manager = multiprocessing.Manager()
        shared_result_dict = manager.dict()

        process_list = []
        for gpu in gpu_devices:
            process = multiprocessing.Process(target=get_embed_process, args=[gpu, shared_result_dict, shared_lock])
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

    logger.info(f"save_code_embed. {len(shared_result_dict)}")
    assert len(shared_result_dict)>0

    try:
        emb_code_dict = {code_id:code_embed for code_id, (code_embed, semester_group, org_token_len, cur_token_len) in shared_result_dict.items()}
        code_embed = CodeEmbed(emb_code_dict)

        with open(os.path.join(embed_save_dir, f"{code_embed_fname}"), 'wb') as pkl_file:
            pickle.dump(code_embed, pkl_file)

        from collections import defaultdict
        semester_group_token_len_dict = defaultdict(list)
        for code_embed, semester_group, org_token_len, cur_token_len in shared_result_dict.values():
            semester_group_token_len_dict[semester_group].append((org_token_len, cur_token_len))

    except Exception as e:
        make_logger.write_exception_log(logger, e)
        exit(-1)

    semester_token_infos = dict()
    for semester_group, token_len_list in semester_group_token_len_dict.items():
        org_token_len_list = [x[0] for x in token_len_list]
        len_mean = float(np.mean(org_token_len_list))
        len_std = float(np.std(org_token_len_list))
        len_min = float(np.min(org_token_len_list))
        len_max = float(np.max(org_token_len_list))

        final_token_len = token_len_list[0][1]
        over_limit_count = 0
        for token_len in org_token_len_list:
            if token_len-4>final_token_len:
                over_limit_count += 1
        
        semester_token_infos[semester_group] = {
            "mean": float(f"{len_mean:.2f}"),
            "std": float(f"{len_std:.2f}"),
            "min": float(f"{len_min:.2f}"),
            "max": float(f"{len_max:.2f}"),
            "total": len(token_len_list),
            "over_limit": over_limit_count,
            "over_ratio": float(f"{over_limit_count/len(token_len_list):4f}")
        }

    import json
    with open(os.path.join(embed_save_dir, f"semester_token_infos.json"), 'w') as f:
        json.dump(semester_token_infos, f, indent=3)

    logger.info("save_code_embed.end")
    return

def load_code_embed(embed_save_dir) -> CodeEmbed:
    with open(os.path.join(embed_save_dir, f"{code_embed_fname}"), 'rb') as pkl_file:
        code_embed = pickle.load(pkl_file)

    return code_embed
