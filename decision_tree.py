import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

DT = DecisionTreeClassifier(criterion='entropy', max_depth=2)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)

fig = plt.figure(figsize=(8,6))
a = plot_tree(DT, fontsize=12, filled=True, 
              class_names= ['0', '1'])
plt.show()