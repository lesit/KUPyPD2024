import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "que_type_models.json"), "r") as f:
    que_type_model_info = json.load(f)

    que_type_models = que_type_model_info["que_type_models"]
    qikt_ab_models = que_type_model_info["qikt_ab_models"]

que_type_models += qikt_ab_models
