#!/bin/bash

nohup python -u kupypd_run.py --devices 0,1 --hp_search > /dev/null 2>&1 &
