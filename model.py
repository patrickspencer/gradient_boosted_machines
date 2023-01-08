#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeRegressor
import numpy as np


class GradientBoostedMachine:
    """Gradient Boosted Machine class"""

    def __init__(self, n_trees, learning_rate, max_depth) -> None:
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.f_zero = 0

    def fit(self, X, y) -> None:
        """Fit method for training the model"""
        self.f_zero = y.mean()
        f_m = self.f_zero
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, y - f_m)
            f_m += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X) -> np.ndarray:
        """Predict method for making predictions"""
        predictions = [t.predict(X) for t in self.trees]
        return self.f_zero + self.learning_rate + np.sum(predictions, axis=0)
