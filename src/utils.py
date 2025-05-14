
try:
    from .dataset_config import *
except Exception as e:
    from dataset_config import *

import pickle
import time

def make_hp_combs(hp_scopes:dict):
    hp_search_comb_list = []
    from itertools import product
    hp_names = hp_scopes.keys()
    hp_value_cand_list = hp_scopes.values()
    for values in product(*hp_value_cand_list):
        params = {k:v for k,v in zip(hp_names, values)}
        hp_search_comb_list.append(params)
    return hp_search_comb_list

def load_attempt_interaction_sequence(dataset_root_dir:str) -> dict:
    """
    return dict
    (semester_group, student_id, problem_id), [[df]]
    df is interactions of a submission attempt
    df.columns is like bellow:
        "type", "code_id", "result", "result_status", 'n_error_feedback', 'error_feedback_ids', 'timestamp'
    """
    st = time.time()

    pkl_path = get_dataset_path(dataset_root_dir, DatasetType.attempt_interaction_sequence_pkl)
    with open(pkl_path, 'rb') as pkl_file:
        attempt_interaction_sequence_dict = pickle.load(pkl_file)

    attempt_interaction_sequence_dfs = dict()
    for key, interaction_sequence_dict in attempt_interaction_sequence_dict.items():
        interaction_columns = interaction_sequence_dict["interaction_columns"]

        interactions_df_list = []
        for attempt_sequence in interaction_sequence_dict["attempt_sequence"]:
            interactions_df = pd.DataFrame(attempt_sequence, columns=interaction_columns)
            interactions_df_list.append(interactions_df)
        
        attempt_interaction_sequence_dfs[key] = interactions_df_list

    print(f"load_attempt_interaction_sequence. elapse: {time.time() - st}")
    return attempt_interaction_sequence_dfs

