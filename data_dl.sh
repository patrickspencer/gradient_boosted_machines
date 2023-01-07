#!/bin/bash
# Download data files


DATA="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
curl $DATA -o test.csv