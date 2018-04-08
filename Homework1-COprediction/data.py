import math
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

df = pd.read_csv('train.csv', header=0,
                 names=['Date', 'Time', 'PT08.S1', 'NMHC', 'C6H6', 'PT08.S2', 'NOx', 'PT08.S3', 'NO2',
                        'PT08.S4', 'PT08.S5', 'T', 'RH', 'AH', 'CO']).set_index(['Date', 'Time'])
df = df.replace(to_replace=-200, value=np.nan)
to_keep = [col for col in df.columns if col not in ['NMHC']]
df = df[to_keep]
ss = StandardScaler()
min_max_scaler = preprocessing.MinMaxScaler()


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
predictors = ['PT08.S3_1', 'C6H6_2', 'NOx_1', 'PT08.S5_1', 'PT08.S1_1',
              'PT08.S2_1', 'C6H6_1', 'CO_1']
df2 = df[['CO'] + predictors]
X_1 = df2[predictors].as_matrix(columns=None)
Y_1 = df2['CO'].as_matrix(columns=None)
y_1 = np.array(Y_1).reshape(4812, 1)
X = min_max_scaler.fit_transform(X_1).tolist()
y = min_max_scaler.fit_transform(y_1).tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

def getdata(x):
    return {
        1 : X_train,
        2 : X_test,
        3 : y_train,
        4 : y_test,
    }[x]