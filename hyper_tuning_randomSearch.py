from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('diabetes.csv')
print(df.head())

X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

rfc = RandomForestClassifier()

parameter_distributions = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'criterion': ['gini', 'entropy']
}

randomSearch = RandomizedSearchCV(rfc, parameter_distributions, random_state=0, n_jobs=-1)
randomSearch.fit(X_train, y_train)
# picking the best model
best_model_rs = randomSearch.best_estimator_
print("best model Random Search\n", best_model_rs)