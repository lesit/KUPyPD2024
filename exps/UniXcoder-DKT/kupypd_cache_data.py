import os
import time
import numpy as np
import pandas as pd

from kupypd_code_emb import *

from kupypd_readdata import DataReader
import kupypd_config

class CacheDataLoad:
    def __init__(self, data_dir, data_aux_path, logger, debug=False):
        self.code_emb = load_code_embed(data_dir)
        
        self.data_aux_df = None
        if data_aux_path is not None:
            df = pd.read_csv(data_aux_path)
            if len(df)>0:
                self.data_aux_df = df

        self.logger = logger
        self.debug = debug

        self.config:kupypd_config.Config = None

        self.cur_data_dir = ""
        self.dkt_features_dir = ""

        self.reader:DataReader = None
        self.train_data:dict = None
        self.np_test_data:np.ndarray = None

    @property
    def code_emb_size(self):
        return self.code_emb.emb_size
    
    def load_data(self, is_hp_search, config:kupypd_config.Config, semester_group, data_dir):
        if self.cur_data_dir==data_dir and self.config is not None:
            if kupypd_config.is_equal_for_data(self.config, config):
                return

        problems_d = np.load(os.path.join(data_dir, "problems.npy"),allow_pickle=True).item()

        self.config = config

        st = time.time()

        self.logger.info(f"load data:{data_dir}. start")
        self.cur_data_dir = data_dir

        if self.data_aux_df is not None and config.solving_interaction_len>0:
            data_aux_df = self.data_aux_df[self.data_aux_df["semester_group"] == semester_group]
            
        else:
            data_aux_df = None

        self.dkt_features_dir = os.path.join(data_dir, "DKTFeatures")
        self.train_data_path = os.path.join(self.dkt_features_dir, f"train_data.csv")
        self.test_data_path = os.path.join(self.dkt_features_dir, f"test_data.csv")
        self.reader = DataReader(config,
                                 problems_d,
                                 data_aux_df,
                                 self.logger,
                                 self.debug)
        
        self.train_data = self.reader.get_data(self.code_emb, self.train_data_path)
        if not is_hp_search:
            self.np_test_data = np.asarray(list(self.reader.get_data(self.code_emb, self.test_data_path).values()))

        self.logger.info(f"load data:{data_dir}. end elapse: {time.time() - st}\n")
    