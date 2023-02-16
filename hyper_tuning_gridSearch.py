import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

df = pd.read_csv('diabetes.csv')
print(df.head())

X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

rfc = RandomForestClassifier()

grid = {
    'max_depth':[3,5,None],
    'n_estimators':[10],
    'max_features':[1,2,3],
    'min_samples_leaf':[1,2,3],
    'min_samples_split':[1,2,3]
}

# Grid Search
from sklearn.model_selection import GridSearchCV

gridCV = GridSearchCV(rfc, param_grid=grid, cv=3, scoring='accuracy')
model_grid = gridCV.fit(X_train,y_train)
print('Best hyperparameters are: '+str(model_grid.best_params_))
print('Best score is: '+str(model_grid.best_score_))