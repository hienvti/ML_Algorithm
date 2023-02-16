import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
dataset = pd.read_csv('diabetes.csv')
print(dataset.head())
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, -1].values

print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = SVC(kernel = 'rbf', random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)
print ('accuracy: ', accuracy_score(y_test, y_pred))