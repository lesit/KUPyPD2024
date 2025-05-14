import os
import time
import numpy as np
import pandas as pd

from kupypd_create_code_ast_path import CodeAstPath
from kupypd_readdata import DataReader
import kupypd_config

class CacheDataLoad:
    def __init__(self, logger, debug=False):
        self.logger = logger
        self.debug = debug

        self.config:kupypd_config.Config = None

        self.cur_data_dir = ""
        self.dkt_features_dir = ""

        self.reader:DataReader = None
        self.train_data:dict = None
        self.np_test_data:np.ndarray = None

    def load_data(self, is_hp_search, config:kupypd_config.Config, data_dir):
        if self.cur_data_dir==data_dir and self.config is not None:
            if kupypd_config.is_equal_for_data(self.config, config):
                return

        self.config = config

        st = time.time()

        self.logger.info(f"load data:{data_dir}. start")
        self.cur_data_dir = data_dir
        self.code_ast_path_inst = CodeAstPath.load(data_dir)
            
        self.dkt_features_dir = os.path.join(data_dir, "DKTFeatures")
        self.train_data_path = os.path.join(self.dkt_features_dir, f"train_data.csv")
        self.test_data_path = os.path.join(self.dkt_features_dir, f"test_data.csv")
        self.reader = DataReader(config,
                                 self.code_ast_path_inst,
                                 self.logger,
                                 self.debug)
        
        self.train_data = self.reader.get_data(self.train_data_path)
        if not is_hp_search:
            self.np_test_data = np.asarray(list(self.reader.get_data(self.test_data_path).values()))

        self.logger.info(f"load data:{data_dir}. end elapse: {time.time() - st}\n")
    