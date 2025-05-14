import numpy as np
import pandas as pd

from kupypd_create_code_ast_path import *
from kupypd_config import Config

class DataReader():
    def __init__(self, config:Config, code_ast_path_inst:CodeAstPath, logger, debug):
        self.config:Config = config

        self.code_ast_path_inst = code_ast_path_inst

        self.numofques = len(code_ast_path_inst.problems_d) # Code-DKT에서는 config에 있는 10으로 고정되어 있었으나, 실제 데이터의 문제 개수를 사용하도록 함

        # Code-DKT에서는 config에 있는 50으로 고정되어 있으나, 이 데이터에서는 문제당 평균 한번 이상에서 표준편자 감안하면 최대 5번의 제출을 한다고 할 수 있다.
        self.entire_maxstep = self.numofques * config.maxstep_per_problem   

        # Skill DKT #1, original + emb_solving_interactions
        self.data_shape = [self.entire_maxstep, 2*self.numofques + 3*self.config.max_code_len]
        self.code_feature_indices = (2*self.numofques, 2*self.numofques + 3*self.config.max_code_len)

        self.logger = logger
        self.debug = debug

    @property
    def ast_node_count(self):
        return self.code_ast_path_inst.node_count
    
    @property
    def ast_path_count(self):
        return self.code_ast_path_inst.path_count
    
    def get_data(self, file_path):
        self.logger.info(f"loading train data... : {file_path}")

        data = dict()
        
        code_df = self.code_ast_path_inst.code_df

        from collections import defaultdict

        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            # if self.debug:
            #     if idx >= 2:
            #         break
                
            # 한 학생에 대한 데이터
            problems = [int(q) for q in row.Problems.strip().split(',')]
            ans = [int(a) for a in row.Result.strip().split(',')]
            css = [cs for cs in row.CodeStates.strip().split(',')]

            step_indices = list(range(len(problems)))

            if len(problems) > self.entire_maxstep:
                if not self.config.cut_by_problem:
                    step_indices = step_indices[-self.entire_maxstep:]
                else:
                    problem_an_cs_dict = defaultdict(list)
                    for problem_idx, problem in enumerate(problems):
                        problem_an_cs_dict[problem].append(problem_idx)

                    for problem, indices in problem_an_cs_dict.items():
                        if len(indices) >= self.config.maxstep_per_problem:
                            step_indices.extend(indices[-self.config.maxstep_per_problem:])
                        else:
                            step_indices.extend(indices)

                    step_indices = sorted(step_indices)

            temp = np.zeros(shape=self.data_shape) 

            extra = self.entire_maxstep - len(step_indices)

            for step_idx, v_idx in enumerate(step_indices):
                c_idx = step_idx + extra
                if ans[v_idx] == 1:
                    temp[c_idx][problems[v_idx]] = 1
                else:
                    temp[c_idx][problems[v_idx] + self.numofques] = 1
                        
                code = code_df[code_df['CodeStateID']==css[v_idx]]['RawASTPath']
                if code.empty:
                    continue    # not parsed because of syntax error, or no ast node because of too simple

                code = code.iloc[0]
                if type(code) == str:
                    code_paths = code.split("@")
                    raw_features = self.code_ast_path_inst.convert_to_idx(self.config, code_paths, self.logger)
                    if len(raw_features) < self.config.max_code_len:
                        raw_features += [[0,0,0]]*(self.config.max_code_len - len(raw_features))    # padding
                    else:
                        raw_features = raw_features[:self.config.max_code_len]

                    features = np.array(raw_features).reshape(-1, self.config.max_code_len*3)
                    temp[c_idx][self.code_feature_indices[0] : self.code_feature_indices[1]] = features

            data[row.student] = temp.tolist()
            
        self.logger.info(f"{len(data)}, {np.asarray(data[list(data.keys())[0]]).shape}")
        return data
