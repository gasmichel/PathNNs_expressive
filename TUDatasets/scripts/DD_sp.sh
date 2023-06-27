#!/bin/sh
python ../main_splits.py --dataset DD --cutoff 2 --path-type shortest_path --residuals;
python ../main_splits.py --dataset DD --cutoff 3 --path-type shortest_path --residuals;
python ../main_splits.py --dataset DD --cutoff 4 --path-type shortest_path --residuals