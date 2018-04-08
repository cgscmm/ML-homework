import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn import metrics

X_train = data.getdata(1)
X_test = data.getdata(2)
y_train = data.getdata(3)
y_test = data.getdata(4)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)

print("Mean Absolute Error: ", mean_absolute_error(y_test, prediction))
print("Median Absolute Error: ", median_absolute_error(y_test, prediction))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, prediction))

# fig, ax = plt.subplots()
# ax.scatter(y_test, prediction)
# #ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()
