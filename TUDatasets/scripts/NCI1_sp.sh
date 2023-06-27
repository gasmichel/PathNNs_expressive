#!/bin/sh
python ../main_splits.py --dataset NCI1 --cutoff 2 --path-type shortest_path --residuals;
python ../main_splits.py --dataset NCI1 --cutoff 3 --path-type shortest_path --residuals;
python ../main_splits.py --dataset NCI1 --cutoff 4 --path-type shortest_path --residuals