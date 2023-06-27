#!/bin/sh
python ../main_splits.py --dataset IMDB-BINARY --cutoff 2 --path-type shortest_path --residuals;
python ../main_splits.py --dataset IMDB-BINARY --cutoff 3 --path-type shortest_path --residuals