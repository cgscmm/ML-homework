# Homework 1
The aim of this homework is using monitored data in the past to predict the CO concentration in the future.
To achieve that, we divide the task into two steps: data preprocessing, and using different algorithms to analysis the data.
### Data preprocessing
First, we use `pandas` to import the `train.csv`:
```python
df = pd.read_csv('train.csv', header=0,
names=['Date', 'Time', 'PT08.S1', 'NMHC', 
'C6H6', 'PT08.S2', 'NOx', 'PT08.S3', 'NO2',
'PT08.S4', 'PT08.S5', 'T', 'RH', 'AH', 'CO']).
set_index(['Date', 'Time'])
```
Here we rename the columns because the brackets in the original file can cause error in the following processing. 

Then, in order to observe the data more clearly, we use `df = df.replace(to_replace=-200, value=np.nan)` to replace the missing value (-200) by `np.nan`. And then use `df.info()` to observe the missing values, the result is
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
We can see that `NMHC` has only 914 valid values, so we just delete this column. Then, we use `dropna(how='any')`  to delete the rows with any missing values.

Next, because we want to do predction, so we need to fitting CO concentration with data in the past. Here we use the data in past three days.
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
...		...
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
Then we choose the variables whose correlation coefficient is larger than 6.4 as the predictor. The data of today cannot be used, so we choose `'PT08.S3_1','C6H6_2','CO_2','NO2_1','NOx_1','PT08.S5_1', 'PT08.S1_1','PT08.S2_1', 'C6H6_1', 'CO_1'` as our predictor.

Finally, we use `sklearn.preprocessing.MinMaxScaler()`method to di the normalization.
### Modeling
#### Linear Regression
First, we use `matplotlib` to show the relationship between the characteristics and CO concentration.
![avatar](https://github.com/cgscmm/ML-homework/blob/master/Homework1-COprediction/data_visualization.png)
We can see that every characteristic has strong linear relation with CO concentration. Thus, we can try to use linear regression method to do the prediction.

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
And the result is 
```python
Mean Absolute Error:  0.0459690487838
Median Absolute Error:  0.0321434214189
Mean Squared Error: 0.00433337386598
```
![img](https://github.com/cgscmm/ML-homework/blob/master/Homework1-COprediction/figure1.jpg)

#### BP network
We use a BP neural network with 9 input nodes, 8 hidden nodes and 1 output nodes. Here we only use bias term in the input layer, because add a bias term in all layers will significantly increase the compute.

And the network we built is totaly based on naive Python without using any toolkit. The code is too large, so we won't do furthur explanation, it is in `BP_network.py`.

**Problems we got**

When build this BP network, we met two problems:

- Slow computing speed
- Poor structure optimization

The naive python has very poor GPU support, so we do all of the matrix calculation by CPU, it will take about 1 minute to update the weights once. And because of my weak coding skill, the structure of codes may be not so elegant. Thus, it takes 2901s to do the whole training and predicting, which is quite a long time.

Finally, the result is:
```python
Mean Absolute Error:  0.0442622577904
Median Absolute Error:  0.0292050137207
Mean Squared Error: 0.00418156528933
```
We can see that the performance of this BP network is better than Linear Regression.


