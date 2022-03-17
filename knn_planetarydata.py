# Beginning of the "knn_planetarydata.py"
"""
Created on Tue Mar 15 19:46:56 2022

This is an offshoot of planetarydataanalysis.py script in the Stat-Analysis repository,
Will use the same initial logic to clean the dataset but then will create a kNN model.
Will create an initial crude model followed by more refined iterations of the model.

@author: José Bravo
"""
# Import necessary libraries
from random import seed
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from scipy import stats
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsRegressor
from  sklearn.metrics import mean_squared_error, r2_score
seed(125)
# Will invoke train-test-split (tts) resampling method.
def tts(data, split = 0.80):
    train = list()
    train_size = split*len(data)
    data_copy = list(data)
    while len(train) < train_size :
        index = randrange(len(data_copy))
        train.append(data_copy.pop(index))
    return np.array(train), np.array(data_copy)

path = 'J:\\datasets\\PS_2022.02.27_18.46.12.csv'
raw_data = pd.read_csv(path)
# Need to remove the first thirteen rows.
# Those rows contain superflous notes/remarks.
raw_data = raw_data.drop(raw_data.index[range(14)])
print(raw_data.columns)
# First data frame with columns of interest
df1 = raw_data.iloc[:, [0, 1, 5]]
df1.reset_index(drop=True, inplace=True)
#df1.reset_index()
df1.rename(columns={'# This file was produced by the NASA Exoplanet Archive  http://exoplanetarchive.ipac.caltech.edu':'planetname', 'Unnamed: 1':'orbitperiod', 'Unnamed: 5':'eccentricity'}, inplace=True)
df1 = df1.drop(df1.index[range(1)])
# Next data frame will ne one with NA values removed
df2 = df1.dropna()
df2.reset_index(drop=True, inplace=True)
# Logic below to determine the data types of the column entries in the latest dataframe.
print(type(df2.iloc[0][1])) # <- Notice the columns are strings and not numeric.
print(type(df2.iloc[0][2]))
# Convert just columns "a" and "b"
df2.loc[:, 'orbitperiod'] = df2.loc[:, 'orbitperiod'].apply(pd.to_numeric)
df2.loc[:, 'eccentricity'] = df2.loc[:, 'eccentricity'].apply(pd.to_numeric)
# Third dataframe
# Want to take averages of multiple planet entries.
x1 = df2.groupby('planetname', as_index = False)['orbitperiod'].mean()
x2 = df2.groupby('planetname', as_index = False)['eccentricity'].mean()
df3 = x1.merge(x2, on=['planetname'])
# Quick distributional plot.
plot1 = plt.hist(df3.orbitperiod, color = 'green', edgecolor = 'black', bins = 45)
plt.title('Distributional Plot')
plt.ylabel('Frequency')
plt.xlabel('Orbital Period')
plt.show(plot1)
# Very skewed distribution
# Split the data into training and testing.
x, y = list(df3.orbitperiod), list(df3.eccentricity)
x_train, x_test = tts(x)
y_train, y_test = tts(y)
x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test)
print('\nTraining x shape: ', x_train.shape)
print('Training y shape: ', y_train.shape)
print('Test x shape: ', x_test.shape)
print('Test y shape: ', y_test.shape)

# Now to build a crude model
# Will run kNN for various values of n_neighbors and store results
knn_r_acc = []
for i in range(1, 17, 1) :
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(x_train, y_train)
    
    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)
    
    knn_r_acc.append((i, test_score, train_score))
    
results = pd.DataFrame(knn_r_acc, columns = ['k', 'Test Score', 'Train Score'])
print(results)
print(results.iloc[:,1].max())
# According to the above, the model yields best fit at k = 15.

model_knn = KNeighborsRegressor(n_neighbors = 15)
model_knn.fit(x_train, y_train)
y_knn_train_pred = model_knn.predict(x_train)
y_knn_test_pred = model_knn.predict(x_test)
# Model Performance
# First, the training mean square error and r2 score.
knn_train_mse = mean_squared_error(y_train, y_knn_train_pred)
knn_train_r2 = r2_score(y_train, y_knn_train_pred)
# Now, test mean square and r2 score.
knn_test_mse = mean_squared_error(y_test, y_knn_test_pred)
knn_test_r2 = r2_score(y_test, y_knn_test_pred)
# Consolidate the results.
knn_results = pd.DataFrame(['k Nearest Neighbor', knn_train_mse, knn_train_r2, knn_test_mse, knn_test_r2]).transpose()
knn_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(knn_results)