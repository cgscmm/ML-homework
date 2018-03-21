# Homework 1
The aim of this homework is using monitored data in this moment to predict the CO concentration in the future.
To achieve that, we can divide the task into two steps, first step is data processing, and the second step is using several statistical models to analysis the data to find a suitable one.
### Data processing
First, we use pandas to import the `train.csv`:
```python
df = pd.read_csv('train.csv', header=0,
names=['Date', 'Time', 'PT08.S1', 'NMHC', 
'C6H6', 'PT08.S2', 'NOx', 'PT08.S3', 'NO2',
'PT08.S4', 'PT08.S5', 'T', 'RH', 'AH', 'CO']).
set_index(['Date', 'Time'])
```
Here I rename the columns because the brackets in the original file can cause error in the following processing. 

Then, in order to observe the data more clearly, I use `df = df.replace(to_replace=-200, value=np.nan)` to replace the missing value (-200) by `np.nan`. And now we can use `df.info()` to observe the missing values, the result is
```python
PT08.S1    7709 non-null float64
NMHC       914 non-null float64
C6H6       7709 non-null float64
PT08.S2    7709 non-null float64
NOx        6394 non-null float64
PT08.S3    7709 non-null float64
NO2        6391 non-null float64
PT08.S4    7709 non-null float64
PT08.S5    7709 non-null float64
T          7709 non-null float64
RH         7709 non-null float64
AH         7709 non-null float64
CO         6344 non-null float64
```
It's easy to recognise that `NMHC` data has only 914 valid values, so we just delete this column. Furthurmore, we use `dropna(how='any')`  to delete the rows with missing values.

Next, because we want to do predction, so we need to fitting CO concentration with data in the PAST. Here we just move the data in upper two rows to that row.
```python
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
            ```
Then the columns are
```python
Index(['PT08.S1', 'C6H6', 'PT08.S2', 'NOx', 'PT08.S3', 'NO2', 'PT08.S4',
       'PT08.S5', 'T', 'RH', 'AH', 'CO', 'PT08.S1_1', 'PT08.S1_2', 'PT08.S1_3',
       'C6H6_1', 'C6H6_2', 'C6H6_3', 'PT08.S2_1', 'PT08.S2_2', 'PT08.S2_3',
       'NOx_1', 'NOx_2', 'NOx_3', 'PT08.S3_1', 'PT08.S3_2', 'PT08.S3_3',
       'NO2_1', 'NO2_2', 'NO2_3', 'PT08.S4_1', 'PT08.S4_2', 'PT08.S4_3',
       'PT08.S5_1', 'PT08.S5_2', 'PT08.S5_3', 'T_1', 'T_2', 'T_3', 'RH_1',
       'RH_2', 'RH_3', 'AH_1', 'AH_2', 'AH_3', 'CO_1', 'CO_2', 'CO_3'],
      dtype='object')
```
 where `CO_1` means the CO concentration of yesterday.
 
 Then we use `corr()` method to calculate the correlation coefficient between these features and CO concentration. The result is
```python
                 CO
PT08.S3   -0.712008
PT08.S3_1 -0.613929
PT08.S3_2 -0.449116
PT08.S3_3 -0.308605
T         -0.113017
T_1       -0.109227
T_2       -0.102330
T_3       -0.098378
AH_3      -0.024506
AH_2      -0.012299
AH_1       0.004381
AH         0.021031
RH_3       0.154036
RH_2       0.165149
RH_1       0.181942
PT08.S4_3  0.193514
RH         0.196294
NO2_3      0.295129
PT08.S4_2  0.333752
PT08.S2_3  0.359228
C6H6_3     0.360549
CO_3       0.375127
PT08.S1_3  0.382557
PT08.S5_3  0.400651
NOx_3      0.419065
NO2_2      0.422765
PT08.S4_1  0.515770
NOx_2      0.534993
PT08.S5_2  0.543417
PT08.S2_2  0.547499
PT08.S1_2  0.548183
C6H6_2     0.548334
CO_2       0.563605
NO2_1      0.578015
PT08.S4    0.624131
NOx_1      0.677436
NO2        0.694108
PT08.S5_1  0.727545
PT08.S1_1  0.759628
NOx        0.775796
PT08.S2_1  0.784546
C6H6_1     0.791578
CO_1       0.818164
PT08.S5    0.857949
PT08.S1    0.874790
PT08.S2    0.913314
C6H6       0.927884
CO         1.000000
```
Then we choose the variables whose correlation coefficient is bigger than 6.4 as the predictor. The data of today cannot be used, so we choose `'PT08.S3_1','C6H6_2','CO_2','NO2_1','NOx_1','PT08.S5_1', 'PT08.S1_1','PT08.S2_1', 'C6H6_1', 'CO_1'` as our predictor.
### Modeling
##### Linear Regression
Here we use `scikit-learn` to help us build the linear regression model
```python
predictors = [ 'PT08.S3_1','C6H6_2','NOx_1','PT08.S5_1', 'PT08.S1_1',
              'PT08.S2_1', 'C6H6_1', 'CO_1']
import statsmodels.api as sm
X = df2[predictors]
y = df2['CO']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)
```

