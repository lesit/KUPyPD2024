#!/bin/bash

nohup python -u kupypd_run.py --devices 0,1 --ablation_study > /dev/null 2>&1 &
