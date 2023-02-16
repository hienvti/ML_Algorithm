import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
# split the data into training/testing sets
X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]

#split the targets into training/testing sets
y_train = diabetes_y[:-20]
y_test = diabetes_y[-20:]

#create linear regression object

li_reg = linear_model.LinearRegression()
li_reg.fit(X_train, y_train)

y_pred = li_reg.predict(X_test)

r_sq = li_reg.score(X_test,y_test)
print(f"coefficient of determination: {r_sq}")

# intercept is wo-bias, is a scalar
print(f"intercept: {li_reg.intercept_}")

# slope is w1, w2, w3, is an array
print(f"slope: {li_reg.coef_}")

print(len(X_test))
print(len(y_test))
plt.scatter(X_test, y_test, color = 'black')
plt.plot(X_test, y_pred, color = 'green', linewidth = 3)
plt.xticks(())
plt.yticks(())

plt.show()