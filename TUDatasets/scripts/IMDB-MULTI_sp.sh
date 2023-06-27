#!/bin/sh
python ../main_splits.py --dataset IMDB-MULTI --cutoff 2 --path-type shortest_path --residuals;
python ../main_splits.py --dataset IMDB-MULTI --cutoff 3 --path-type shortest_path --residuals