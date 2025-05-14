#!/bin/bash

nohup python -u kupypd_eval.py --devices 0,1 --train_saved_dir kupypd_result/train > /dev/null 2>&1 &
