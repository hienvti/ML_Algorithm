from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('diabetes.csv')
print(data.head())
X = data.iloc[:, 0:8].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
model.fit(X_train)

distance, indices = model.kneighbors(X_train)
print('indices: ', indices)
print('distance: ', distance)

print('kneighbor graph:\n', model.kneighbors_graph(X_train, mode='distance'))