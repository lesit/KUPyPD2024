import enum
class solving_interaction_types(enum.Enum):
    none = 0
    simple = enum.auto()
    result_status = enum.auto()

class Config:
    def __init__(self):
        self.lr = 0.005
        self.batch_size = 32
        self.epochs = 200

        self.solving_interactions = solving_interaction_types.none
        self.solving_num_embeddings = 0
        self.solving_interaction_len = 0
        self.emb_solving_interaction_size = 0

        self.hidden = 128
        self.layers = 1

        # about readdata
        self.maxstep_per_problem = 4

        self.cut_by_problem = False

config_keys_for_data = ["maxstep_per_problem", "cut_by_problem", "solving_num_embeddings", "solving_interaction_len"]

def is_equal_for_data(src:Config, dst:Config):
    for key in config_keys_for_data:
        src_value = src.__getattribute__(key)
        dst_value = dst.__getattribute__(key)
        if src_value != dst_value:
            return False
    return True
