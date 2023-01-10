#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DataLoader:
    """Class to load data"""

    def __init__(self) -> None:
        self.data = datasets.load_breast_cancer()

        self.df = pd.DataFrame(
            self.data.data,
            columns=self.data.feature_names
        )
        self.X = self.data.data
        self.y = self.data.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, random_state=0)
