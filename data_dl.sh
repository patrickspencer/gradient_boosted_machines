#!/bin/bash
# Download data files


DATA="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
curl $DATA -o bc_data.csv

NAMES="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.names"
curl $NAMES -o bc_names.csv