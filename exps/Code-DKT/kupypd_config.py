class Config:
    def __init__(self):
        self.lr = 0.005
        self.batch_size = 32
        self.epochs = 200

        self.emb_size = 50
        self.hidden = 128
        self.layers = 1

        self.max_code_len = 100

        # about readdata
        self.code_path_length = 8
        self.code_path_width = 2

        self.maxstep_per_problem = 4

        self.cut_by_problem = False

        self.abandon_low_frequence = 0

config_keys_for_data = ["code_path_length", "code_path_width", "maxstep_per_problem", "cut_by_problem", "abandon_low_frequence"]

def is_equal_for_data(src:Config, dst:Config):
    for key in config_keys_for_data:
        src_value = src.__getattribute__(key)
        dst_value = dst.__getattribute__(key)
        if src_value != dst_value:
            return False
    return True
