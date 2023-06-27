#!/bin/sh
python ../main_splits.py --dataset PROTEINS --cutoff 2 --path-type shortest_path --residuals;
python ../main_splits.py --dataset PROTEINS --cutoff 3 --path-type shortest_path --residuals;
python ../main_splits.py --dataset PROTEINS --cutoff 4 --path-type shortest_path --residuals