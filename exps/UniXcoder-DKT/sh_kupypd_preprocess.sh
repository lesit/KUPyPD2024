#!/bin/bash

nohup python -u kupypd_preprocess.py --devices 0,1 > /dev/null 2>&1 &
