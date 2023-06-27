#!/bin/sh
python ../main_splits.py --dataset NCI1 --cutoff 3 --path-type all_simple_paths --residuals;
python ../main_splits.py --dataset NCI1 --cutoff 4 --path-type all_simple_paths --residuals