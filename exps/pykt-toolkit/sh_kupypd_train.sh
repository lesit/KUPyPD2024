#!/bin/bash

nohup python -u kupypd_train.py --devices 0,1 --model_param_jsons model_params_default.json  --hp_tunes hp_tunes.json > /dev/null 2>&1 &
