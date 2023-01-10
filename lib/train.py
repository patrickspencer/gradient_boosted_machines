#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib.data import DataLoader
from model import GradientBoostedMachine


def train(X, y):
    """Train method"""
    clf = GradientBoostedMachine(n_trees=100, learning_rate=0.1, max_depth=3)
    clf.fit(X, y)
    return clf


if __name__ == "__main__":
    data = DataLoader()
    train(data.X_train, data.y_train)
