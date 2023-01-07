from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

data = datasets.load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

learning_rate = 0.3
n_trees = 10
max_depth = 100

F0 = y.mean()
Fm = F0
trees = []

for _ in range(n_trees):
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X_train, y_train - Fm)
    Fm += learning_rate * tree.predict(X_train)
    trees.append(tree)

trees_predict = [t.predict(X_train) for t in trees]
y_hat = F0 + learning_rate * np.sum(trees_predict, axis=0)
