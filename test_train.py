#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train import train
from data import DataLoader
from model import GradientBoostedMachine


def test_data_loader():
    """Test data loader"""
    data = DataLoader()
    assert data.X_train.size != 0
    assert data.X_test.size != 0
    assert data.y_train.size != 0
    assert data.y_test.size != 0


def test_clf():
    """Test gradient boosted machine classifer loaded"""
    data = DataLoader()
    clf = train(data.X_train, data.y_train)
    assert isinstance(clf, GradientBoostedMachine)


if __name__ == "__main__":
    test_data_loader()
    test_clf()
    print("All tests passed")
