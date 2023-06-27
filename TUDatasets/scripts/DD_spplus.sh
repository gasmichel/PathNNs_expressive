#!/bin/sh
python ../main_splits.py --dataset DD --cutoff 3 --path-type all_shortest_paths --residuals;
python ../main_splits.py --dataset DD --cutoff 4 --path-type all_shortest_paths --residuals