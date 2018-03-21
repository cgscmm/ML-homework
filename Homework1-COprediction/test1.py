import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv', header=0,
                 names=['Date', 'Time', 'PT08.S1', 'NMHC', 'C6H6', 'PT08.S2', 'NOx', 'PT08.S3', 'NO2',
                        'PT08.S4', 'PT08.S5', 'T', 'RH', 'AH', 'CO']).set_index(['Date', 'Time'])

to_keep = [col for col in df.columns if col not in ['NMHC']]
df = df[to_keep]
df = df.replace(to_replace=-200, value=np.nan)
# _ = df.fillna(
#     {'PT08.S1': 1098.177336, 'C6H6': 10.458361, 'PT08.S2': 953.451183, 'NOx': 236.664154, 'PT08.S3': 848.642723,
#      'NO2': 106.793694, 'PT08.S4': 1507.230315, 'PT08.S5': 1024.074696, 'T': 19.474614, 'RH': 48.967444, 'AH': 1.082099,
#      'CO': 2.185735}, inplace=True)


def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


for feature in ['PT08.S1', 'C6H6', 'PT08.S2', 'NOx', 'PT08.S3', 'NO2',
                'PT08.S4', 'PT08.S5', 'T', 'RH', 'AH', 'CO']:
    if feature != 'Time' and 'Date':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)
df = df.dropna(how='any')
print(df.columns)
#
#
# spread = df.describe().T
# # precalculate interquartile range for ease of use in next calculation
# IQR = spread['75%'] - spread['25%']
#
# # create an outliers column which is either 3 IQRs below the first quartile or
# # 3 IQRs above the third quartile
# spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))
#
# # just display the features containing extreme outliers
# spread.ix[spread.outliers,]
# print(spread)
# df.info()

# plt.rcParams['figure.figsize'] = [14, 8]
# df.NMHC.hist()
# plt.title('Distribution of NMHC(GT)')
# plt.xlabel('NMHC(GT)')
# plt.show()

####################################################################################################
# print(df.corr()[['CO']].sort_values('CO'))
#                  CO
# PT08.S3   -0.712008
# PT08.S3_1 -0.613929
# PT08.S3_2 -0.449116
# PT08.S3_3 -0.308605
# T         -0.113017
# T_1       -0.109227
# T_2       -0.102330
# T_3       -0.098378
# AH_3      -0.024506
# AH_2      -0.012299
# AH_1       0.004381
# AH         0.021031
# RH_3       0.154036
# RH_2       0.165149
# RH_1       0.181942
# PT08.S4_3  0.193514
# RH         0.196294
# NO2_3      0.295129
# PT08.S4_2  0.333752
# PT08.S2_3  0.359228
# C6H6_3     0.360549
# CO_3       0.375127
# PT08.S1_3  0.382557
# PT08.S5_3  0.400651
# NOx_3      0.419065
# NO2_2      0.422765
# PT08.S4_1  0.515770
# NOx_2      0.534993
# PT08.S5_2  0.543417
# PT08.S2_2  0.547499
# PT08.S1_2  0.548183
# C6H6_2     0.548334
# CO_2       0.563605
# NO2_1      0.578015
# PT08.S4    0.624131
# NOx_1      0.677436
# NO2        0.694108
# PT08.S5_1  0.727545
# PT08.S1_1  0.759628
# NOx        0.775796
# PT08.S2_1  0.784546
# C6H6_1     0.791578
# CO_1       0.818164
# PT08.S5    0.857949
# PT08.S1    0.874790
# PT08.S2    0.913314
# C6H6       0.927884
# CO         1.000000
predictors = [ 'CO_2','NO2_1','NOx_1','PT08.S5_1', 'PT08.S1_1',
              'PT08.S2_1', 'C6H6_1', 'CO_1']
df2 = df[['CO'] + predictors]
# # manually set the parameters of the figure to and appropriate size
# plt.rcParams['figure.figsize'] = [16, 22]
# # call subplots specifying the grid structure we desire and that
# # the y axes should be shared
# fig, axes = plt.subplots(nrows=4, ncols=2, sharey=True)
#
# # Since it would be nice to loop through the features in to build this plot
# # let us rearrange our data into a 2D array of 6 rows and 3 columns
# arr = np.array(predictors).reshape(4, 2)
#
# # use enumerate to loop over the arr 2D array of rows and columns
# # and create scatter plots of each meantempm vs each feature
# for row, col_arr in enumerate(arr):
#     for col, feature in enumerate(col_arr):
#         axes[row, col].scatter(df2[feature], df2['CO'])
#         if col == 0:
#             axes[row, col].set(xlabel=feature, ylabel='CO')
#         else:
#             axes[row, col].set(xlabel=feature)
import statsmodels.api as sm
X = df2[predictors]
y = df2['CO']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
# instantiate the regressor class
regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn import metrics
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))
print ("MSE:",metrics.mean_squared_error(y_test, prediction))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, prediction)))
fig, ax = plt.subplots()
ax.scatter(y_test, prediction)
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()