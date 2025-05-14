import os
import pandas as pd
import numpy as np
import pickle
from kupypd_config import Config

def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

class CodeAstPath():
    def __init__(self, problems_d:dict, code_df:pd.DataFrame):
        self.problems_d:dict = problems_d
        self.code_df:pd.DataFrame = code_df
        self.node_hist:dict = None
        self.path_hist:dict = None
        self.node_word_index:dict = None
        self.path_word_index:dict = None

    @property
    def node_count(self):
        return len(self.node_hist)
    
    @property
    def path_count(self):
        return len(self.path_hist)

    def create(self, training_students, logger):
        # 학습에 사용된 코드에 대해서만 word index를 구해서, test 시에는 학습에 대해서 평가할 수 있도록 한다
        all_training_code = self.code_df[self.code_df['student_id'].isin(training_students)]['RawASTPath']

        logger.info("CodeAstPath.split codes")
        separated_code = []
        for code in all_training_code:
            if type(code) == str:
                node_list = code.split("@")
                if len(node_list[0])==0:
                    a = 0
                separated_code.append(node_list)
        
        logger.info("CodeAstPath.make code and node path")
        node_hist = {}
        path_hist = {}
        for idx, paths in enumerate(separated_code):
            try:
                starting_nodes = [p.split(",")[0] for p in paths]
                path = [p.split(",")[1] for p in paths]
                ending_nodes = [p.split(",")[2] for p in paths]
            except Exception as e:
                logger.info(f"exception:{str(e)}")
                exit(-1)
                
            nodes = starting_nodes + ending_nodes
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1

        # small frequency then abandon, for node and path
        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        # create ixtoword and wordtoix lists
        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)

        self.node_hist = node_hist
        self.path_hist = path_hist
        self.node_word_index = node_word_index
        self.node_index_word = node_index_word
        self.path_word_index = path_word_index
        self.path_index_word = path_index_word

    def convert_to_idx(self, config:Config, sample, logger):
        """
        Converting to the index 
        Input:
        sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
        node_word_index: dict. The node to word index dictionary.
        path_word_index: dict. The path to word index dictionary.

        """
        unk_node_word_idx = self.node_word_index['UNK']
        unk_path_word_idx = self.path_word_index['UNK']

        # skipped_low_freqs = []
        sample_index = []
        for line in sample:
            components = line.split(",")

            path_word = components[1]

            if config.abandon_low_frequence>0:
                path = self.path_word_index.get(path_word)
                if path is None:
                    continue

                path_freq = self.path_hist[path_word]
                if path_freq <= config.abandon_low_frequence:
                    # skipped_low_freqs.append(path_freq)
                    continue
            else:
                path = self.path_word_index.get(path_word, unk_path_word_idx)

            starting_node = self.node_word_index.get(components[0], unk_node_word_idx)
            ending_node = self.node_word_index.get(components[2], unk_node_word_idx)

            sample_index.append([starting_node, path, ending_node])

        # if len(skipped_low_freqs)>0:
        #     logger.info(f"convert_to_idx.skipped_low_freqs: {len(skipped_low_freqs)}")
        return sample_index

    @staticmethod
    def get_save_path(data_dir):
        return os.path.join(data_dir, "code_ast_path.pkl")
    
    @staticmethod
    def save(save_dir, logger):
        problems_d = np.load(os.path.join(save_dir, "problems.npy"),allow_pickle=True).item()
        code_df = pd.read_csv(os.path.join(save_dir, "labeled_paths.tsv"),sep="\t")
        training_students = np.load(os.path.join(save_dir, "training_students.npy"),allow_pickle=True)

        code_ast_path_inst = CodeAstPath(problems_d, code_df)
        code_ast_path_inst.create(training_students, logger)

        save_path = CodeAstPath.get_save_path(save_dir)
        with open(save_path, 'wb') as pkl_file:
            pickle.dump(code_ast_path_inst, pkl_file)

            logger.info(f"save_code_node_path. save to {save_path}")

    @staticmethod
    def load(data_dir):
        saved_path = CodeAstPath.get_save_path(data_dir)
        with open(saved_path, 'rb') as pkl_file:
            code_ast_path_inst = pickle.load(pkl_file)
        
        return code_ast_path_inst
