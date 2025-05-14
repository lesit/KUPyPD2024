import numpy as np
import pandas as pd

from kupypd_config import Config, solving_interaction_types
from kupypd_code_emb import CodeEmbed

class DataReader():
    def __init__(self, config:Config, problems_d, data_aux_df:pd.DataFrame, logger, debug):
        self.config:Config = config

        self.data_aux_df = None
        self.solving_interaction_len = 0
        if config.solving_interactions!=solving_interaction_types.none and config.solving_interaction_len>0:
            assert data_aux_df is not None
            if data_aux_df is not None:
                self.data_aux_df = data_aux_df
                self.solving_interaction_len = config.solving_interaction_len

                if config.solving_interactions == solving_interaction_types.simple:
                    data_aux_df["encodes"] = self.data_aux_df.simple_encodes.fillna("")
                elif config.solving_interactions == solving_interaction_types.result_status:
                    data_aux_df["encodes"] = self.data_aux_df.status_represented_encodes.fillna("")
                
        self.numofques = len(problems_d) # Code-DKT에서는 config에 있는 10으로 고정되어 있었으나, 실제 데이터의 문제 개수를 사용하도록 함

        # Code-DKT에서는 config에 있는 50으로 고정되어 있으나, 이 데이터에서는 문제당 평균 한번 이상에서 표준편자 감안하면 최대 5번의 제출을 한다고 할 수 있다.
        self.entire_maxstep = self.numofques * config.maxstep_per_problem   

        self.logger = logger
        self.debug = debug

    def get_solving_interactions(self, code_id):
        row = self.data_aux_df[self.data_aux_df["submission_id"] == code_id].iloc[0]
        solving_interactions = row.encodes
        if len(solving_interactions) > 0:
            solving_interactions = solving_interactions.split(',')
            try:
                solving_interactions = [int(x) for x in solving_interactions]
            except Exception as e:
                exit(-1)
        else:
            solving_interactions = []
        if len(solving_interactions)>self.solving_interaction_len:
            solving_interactions = solving_interactions[-self.solving_interaction_len:]
        else:
            pad = [0 for x in range(self.solving_interaction_len - len(solving_interactions))]
            solving_interactions = pad + solving_interactions

        return solving_interactions
        
    def get_data(self, code_emb_inst:CodeEmbed, file_path):
        self.logger.info(f"loading train data... : {file_path}")

        # Skill DKT #1, original + emb_solving_interactions
        self.data_shape = [self.entire_maxstep, 2*self.numofques + code_emb_inst.emb_size + self.solving_interaction_len]
        self.code_feature_indices = (2*self.numofques, 2*self.numofques + code_emb_inst.emb_size)

        data = dict()
        
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
                        
                code_emb = code_emb_inst.get_code_emb(css[v_idx])
                temp[c_idx][self.code_feature_indices[0] : self.code_feature_indices[1]] = code_emb

                if self.solving_interaction_len>0:
                    solving_interactions = self.get_solving_interactions(css[v_idx])
                    temp[c_idx][self.code_feature_indices[1]:] = solving_interactions

            data[row.student] = temp.tolist()
            
        self.logger.info(f"{len(data)}, {np.asarray(data[list(data.keys())[0]]).shape}")
        return data
