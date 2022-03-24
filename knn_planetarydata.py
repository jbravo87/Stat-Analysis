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
import seaborn as sns
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
# from numpy import mean
# from numpy import std
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# # Another approach to evaluate the model.
# cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
# n_scores = cross_val_score(model_knn, x, y, scoring = 'accuracy', cv = cv, n_jobs = -1, error_score = 'raise')
# # Report the model performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# In this part want to explicitly declare the median and IQR of the
# two columns which are the variables.
import statistics
ecce = df3['eccentricity']
orbper = df3['orbitperiod']
eccen_med = statistics.median(list(ecce))
orbper_med = statistics.median(list(orbper))
print('\nThe median of the orbital period: %.2f' % orbper_med)
print('\nThe median of the eccentricity: %.2f' % eccen_med)
iqr_ecce = stats.iqr(ecce, interpolation = 'midpoint')
iqr_op = stats.iqr(orbper, interpolation = 'midpoint')
print('\nThe interquartile range of the eccentricity: %.2f' % iqr_ecce)
print('\nThe interquartile range of the orbital period: %.2f' % iqr_op)
# Going to store the scaled data in a for orbit period and b for eccentricity
a = (orbper - np.median(orbper))/iqr_op
b = (ecce - np.median(ecce))/iqr_ecce
x3 = df3[['orbitperiod','eccentricity']] # <- this is a dataframe
from sklearn.preprocessing import RobustScaler
rs = RobustScaler().fit(x3)
print(rs.transform(x3))
transformer = rs.transform(x3)
print('\nThe maximum orbital period is: %.2f ' % orbper.max())
# Want to perform Spearman correlation coefficient calculation
spearman_coeff = stats.spearmanr(ecce, orbper)
print(spearman_coeff)
tau_test = stats.kendalltau(ecce, orbper)
print('The Kendall Tau Test Results: ', tau_test)

# Now want to create second knn model using the data that had been treated through robust scaling
#a, b = pd.DataFrame(a), pd.DataFrame(b)
c = [a, b]
df4 = pd.concat(c, axis=1)
# Split the data into training and testing.
x2, y2 = list(df4.orbitperiod), list(df4.eccentricity)
x2_train, x2_test = tts(x2)
y2_train, y2_test = tts(y2)
x2_train, y2_train = pd.DataFrame(x2_train), pd.DataFrame(y2_train)
x2_test, y2_test = pd.DataFrame(x2_test), pd.DataFrame(y2_test)
print('\nTraining x2 shape: ', x2_train.shape)
print('Training y2 shape: ', y2_train.shape)
print('Test x2 shape: ', x2_test.shape)
print('Test y2 shape: ', y2_test.shape)
# Will run kNN for various values of n_neighbors and store results
knn_r_acc2 = []
for j in range(1, 17, 1) :
    knn2 = KNeighborsRegressor(n_neighbors = j)
    knn2.fit(x2_train, y2_train)
    
    test_score2 = knn.score(x2_test, y2_test)
    train_score2 = knn.score(x2_train, y2_train)
    
    knn_r_acc2.append((j, test_score2, train_score2))
    
results2 = pd.DataFrame(knn_r_acc2, columns = ['k', 'Test Score', 'Train Score'])
print(results2)
print(results2.iloc[:,1].max())
# According to the above, the model yields best fit at k = 15, also.
model_knn2 = KNeighborsRegressor(n_neighbors = 15)
model_knn2.fit(x2_train, y2_train)
y_knn_train_pred2 = model_knn2.predict(x2_train)
y_knn_test_pred2 = model_knn2.predict(x2_test)
# Model Performance
# First, the training mean square error and r2 score.
knn_train_mse2 = mean_squared_error(y2_train, y_knn_train_pred2)
knn_train_r2_2 = r2_score(y2_train, y_knn_train_pred2)
# Now, test mean square and r2 score.
knn_test_mse2 = mean_squared_error(y2_test, y_knn_test_pred2)
knn_test_r2_2 = r2_score(y2_test, y_knn_test_pred2)
# Consolidate the results.
knn_results2 = pd.DataFrame(['k Nearest Neighbor', knn_train_mse2, knn_train_r2_2, knn_test_mse2, knn_test_r2_2]).transpose()
knn_results2.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(knn_results2)

def box_plots(df):
    plt.figure(figsize = (10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()
box_plots(df4.orbitperiod)

df3.plot(kind="box", subplots=True, layout=(7,2), figsize=(15,20));
# Function to identify the outliers using the IQR method
def iqr_outlier(x, factor) :
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    min_ = q1 - factor * iqr
    max_ = q3 + factor * iqr
    result_ = pd.Series([0] * len(x))
    result_[((x < min_) | (x > max_))] = 1
    return result_

# Scatter plots highlighting outliers calculated using IQR method.
fig, ax = plt.subplots(7, 2, figsize=(20, 30))
row = col = 0
for n, m in enumerate(df4.columns) :
    if (n % 2 ==0) & (n > 0) :
        row += 1
        col = 0
    outliers = iqr_outlier(df3[m], 1.5)
        
    if sum(outliers) == 0 :
        sns.scatterplot( x = np.arrange(len(df4[m])), y = df4[m], ax = ax[row, col], legend=False, color = 'green')
    else:
        sns.scatterplot(x = np.arange(len(df3[m])), y = df4[m], ax = ax[row, col], hue = outliers, palette = ['green', 'red'])
    for x, y in zip(np.arange(len(df4[m]))[outliers == 1], df4[m][outliers == 1]) :
        ax[row, col].text(x = x, y = y, s = y, fontsize = 8)
    ax[row, col].set_ylabel("")
    ax[row, col].set_title(m)
    ax[row, col].xaxis.set_visible(False)
    if sum(outliers) > 0:
        ax[row, col].legend(ncol = 2)
    col += 1
#ax[row, col].axis('off')
plt.show()
    
# Calculating the Euclidean distance of the data set to detect outliers.
def euclidean_distance_outliers(x, cutoff) :
    result_ = pd.Series([0]*len(x))
    data_mean = x.mean() # mean of data
    dist = np.sqrt(np.sum((x - data_mean)**2)) # Euclidean distance
    dist_mean = dist.mean() # Mean of the distances
    dist_zscore = np.abs((dist - dist_mean)/dist.std()) # z-score of the distances
    result_[((dist_zscore > cutoff))] = 1
    return result_

# euc_d = df3[['orbitperiod', 'eccentricity']].copy()
# euc_d['outlier'] = euclidean_distance_outliers(euc_d, 3)
# sns.scatterplot(x = 'eccentricity', y = 'orbitperiod', data = euc_d, hue = "outlier", palette = ['green', 'red'])

z1 = iqr_outlier(df3.orbitperiod, 1.5)
z1 = pd.DataFrame(z1)
z2 = iqr_outlier(df3.eccentricity, 1.5)
z2 = pd.DataFrame(z2)
z3 = [df3, z1, z2]
#z4 = [df3, z2]
df5 = pd.concat(z3, axis = 1)
#df5 = pd.concat(z4, axis = 1)
print(type(z1)) # Type is Pandas Series
#df5.rename(columns = {0 : 'orbper_outlr'}, inplace = True)
df5.columns = ['planetnames', 'orbitperiod', 'eccentricity', 'orbper_outlr', 'ecce_outlr']
#type(df5.orbper_outlr[300])
# Above is an integer.
df5.size
# Will establish new boolean variable
in_iqr1 = df5['orbper_outlr'] == 0
df6 = df5[in_iqr1]
# v1 <- value 1, v2 <- value 2
v1 = len(df5)
v2 = len(df6)
prcnt_diff = ((v1 - v2)/((v1+v2)/2))*100
print('\nThe percent difference from df5 to df6 is: %.1f' % prcnt_diff)
in_iqr2 = df6['ecce_outlr'] == 0
df7 = df6[in_iqr2]
x4 = df7[['orbitperiod','eccentricity']] # <- this is a dataframe
rs2 = RobustScaler().fit(x4)
print(rs2.transform(x4))
transformer2 = rs2.transform(x4)
transformer2 = pd.DataFrame(transformer2)
x4.columns = ['orbitperiod', 'eccentricity']
transformer2.columns = ['orbitperiod', 'eccentricity']