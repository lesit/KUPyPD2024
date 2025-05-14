#!/bin/bash

nohup python -u kupypd_run.py --devices 0,1 --solving_interactions result_status > /dev/null 2>&1 &
