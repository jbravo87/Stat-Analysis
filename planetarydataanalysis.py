# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:34:51 2022

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

seed(1234)
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
sample_df = df2.iloc[0:789]
#######
# Notice that the data type for the last two columns 
# is string and not double or numeric
#######
print(type(sample_df.iloc[0][1]))
print(type(sample_df.iloc[0][2]))
# Convert just columns "a" and "b"
sample_df.loc[:, 'orbitperiod'] = sample_df.loc[:, 'orbitperiod'].apply(pd.to_numeric)
sample_df.loc[:, 'eccentricity'] = sample_df.loc[:, 'eccentricity'].apply(pd.to_numeric)

# Function to get the mean of a column from a dataframe
def calc_mean(x_dataframe, column):
    planetnames_list = list(set(x_dataframe['planetname'].values.tolist()))
    header = list(x_dataframe.columns.values)
    header.remove(column)
    new_header = header
    new_df = pd.DataFrame(columns = new_header)
    for i,planet in enumerate(planetnames_list):
        results = x_dataframe[(x_dataframe['planetname'] == planet)].describe()
        #results = x_dataframe[(x_dataframe['planetname'] == planet)]
        new_row = []
        for column in new_header:
            if column in ['planetname']:
                new_row.append(planet)
                continue
            new_row.append(results[column].mean)
        new_df.loc[i] = new_row
    return new_df
# Third dataframe
x1 = sample_df.groupby('planetname', as_index = False)['orbitperiod'].mean()
x2 = sample_df.groupby('planetname', as_index = False)['eccentricity'].mean()
df3 = x1.merge(x2, on=['planetname'])
# Distributional plots
# Want to visually see if the distribution is symmetric or not.
# plot1 = sns.displot(df3.orbitperiod, color = 'green')
plot1 = plt.hist(df3.orbitperiod, color = 'green', edgecolor = 'darkorange', bins=25)
plt.title("Distributional Plot")
plt.ylabel('Frequency')
plt.xlabel('Orbital Period')
plt.show(plot1)
#plot2 = sns.displot(df3.eccentricity, color = 'darkorange')
plot2 = plt.hist(df3.eccentricity, color = 'darkorange', edgecolor = 'black', bins = 50)
plt.title("Distributional Plot")
plt.ylabel("Frequency")
plt.xlabel("Eccentricity")
plt.show(plot2)
# Function to find the interquartile range
def iqr(some_array, column):
    percentile25 = some_array[column].quantile(0.25)
    percentile75 = some_array[column].quantile(0.75)
    # Interquartile Range
    iqr = percentile75 - percentile25
    # Lower Limit
    infimum = percentile25 - 1.5*iqr
    # Upper Limit
    supremum = percentile75 + 1.5*iqr
    # Finding outliers
    some_array[some_array[column] > supremum]
    some_array[some_array[column] < infimum]
    # Trimming
    new_array = some_array[some_array[column] < supremum]
    return new_array

# Fourth Dataframe
df4 = iqr(df3, 'eccentricity')
df4 = iqr(df3, 'orbitperiod')
print('Size of the fourth (final) data frame: ', df4.shape)

# To get the number of rows/observations
df4.shape[0]

# More distributional plots
#plot3 = plt.hist(df4.orbitperiod, color = "olivedrab", density = True, edgecolor = 'black', bins=20)
#plt.title("Histrogram of Orbit Period column")
#plt.show(plot3)

# Now logic for the bootstrapping method.
# First, the orbit period column .
orbitperiod = df4.iloc[:, 1]
results1 = []
for nrepeat in range(1000):
    sample = resample(orbitperiod)
    results1.append(sample.median())
results1 = pd.Series(results1)
print('\nBootstrap Statistics for orbit period column: \n')
print(f'original: {orbitperiod.median()}')
print(f'bias: {results1.mean() - orbitperiod.median()}')
print(f'std. error: {results1.std()}')

print("\nOrbit period mean: %.4f" % orbitperiod.mean())
#np.random.seed(seed = 3)
#np.random.seed(seed = 14)
# Create a sample of 20 orbit period data
sample20_orbper = resample(orbitperiod, n_samples = 20, replace = False)
print('Sample of 20 mean: %.4f' % sample20_orbper.mean())
results3 = []
for nrepeat in range(500) :
    sample = resample(sample20_orbper)
    results3.append(sample.mean())
results3 = pd.Series(results3)
# Confidence Interval for Orbital Period
confidence_interval = list(results3.quantile([0.05, 0.95]))
ax = results3.plot.hist(bins=30, figsize=(4, 3), color='coral')
ax.plot(confidence_interval, [55, 55], color='black')
for i in confidence_interval:
    ax.plot([i, i], [0, 65], color='black')
    ax.text(i, 70, f'{i:.0f}', 
            horizontalalignment='center', verticalalignment='center')
ax.text(sum(confidence_interval) / 2, 60, '90% interval',
        horizontalalignment='center', verticalalignment='center')

mean_orbitperiod = results3.mean()
ax.plot([mean_orbitperiod, mean_orbitperiod], [0, 50], color='black', linestyle='--')
ax.text(mean_orbitperiod, 10, f'Mean: {mean_orbitperiod:.0f}',
        bbox=dict(facecolor='white', edgecolor='white', alpha=0.5),
        horizontalalignment='center', verticalalignment='center')
ax.set_ylim(0, 80)
ax.set_ylabel('Counts')

plt.tight_layout()
plt.title('Confidence Interval for Orbit Period')
plt.show()
# Now, the eccentricity column.
eccen = df4.iloc[:, 2]
results2 = []
for nrepeat in range(1000):
    sample = resample(eccen)
    results2.append(sample.median())
results2 = pd.Series(results2)
print('\nBootstrap Statistics for eccentricity column: \n')
print(f'original: {eccen.median()}')
print(f'bias: {results2.mean() - eccen.median()}')
print(f'std. error: {results2.std()}')
print("\nOrbit period mean: %.4f" % eccen.mean())
# Create a sample of 20 orbit period data
sample20_eccen = resample(eccen, n_samples = 20, replace = False)
print('Sample of 20 mean: %.4f' % sample20_eccen.mean())
results4 = []
for nrepeat2 in range(500) :
    sample2 = resample(sample20_eccen)
    results4.append(sample2.mean())
results4 = pd.Series(results4)
# Confidence Interval for Orbital Period
confidence_interval2 = list(results4.quantile([0.05, 0.95]))
ax2 = results4.plot.hist(bins=30, figsize=(4, 3), color='indigo')
ax2.plot(confidence_interval2, [55, 55], color='black')
for j in confidence_interval2:
    ax2.plot([j, j], [0, 65], color='black')
    ax2.text(j, 70, f'{j: .3f}',  
            horizontalalignment='center', verticalalignment='center')
ax2.text(sum(confidence_interval2) / 2, 60, '90% interval',
        horizontalalignment='center', verticalalignment='center')
mean_eccen = results4.mean()
ax2.plot([mean_eccen, mean_eccen], [0, 50], color='black', linestyle='--')
ax2.text(mean_eccen, 10, f'Mean: {mean_eccen: .3f}',
        bbox=dict(facecolor='white', edgecolor='white', alpha=0.5),
        horizontalalignment='center', verticalalignment='center')
ax2.set_ylim(0, 80)
ax2.set_ylabel('Counts')

plt.tight_layout()
plt.title('Confidence Interval for Eccentricity')
plt.show()

# Plot4 <- quick visualization of x-y relationship.
plot4 = plt.scatter([df4.orbitperiod], [df4.eccentricity], color = 'darkcyan')
plt.xlabel("Orbital Period")
plt.ylabel("Eccentricity")
plt.title("Scatterplot")
plt.show(plot4)

# Will invoke train-test-split (tts) resampling method.
def tts(data, split = 0.80):
    train = list()
    train_size = split*len(data)
    data_copy = list(data)
    while len(train) < train_size :
        index = randrange(len(data_copy))
        train.append(data_copy.pop(index))
    return train, data_copy

# Extract variables of interest.
X, y = list(df4.orbitperiod), list(df4.eccentricity)
X_train, X_test = tts(X)
y_train, y_test = tts(y)
X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
X_test, y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)
print('\nTraining X shape: ', X_train.shape)
print('Training y shape: ', y_train.shape)
print('Test X shape: ', X_test.shape)
print('Test y shape: ', y_test.shape)

# Build the model.
from sklearn.ensemble import RandomForestRegressor
randfor = RandomForestRegressor(max_depth = 2, random_state = 42)
randfor.fit(X_train, y_train)

y_rf_train_pred = randfor.predict(X_train)
y_rf_test_pred = randfor.predict(X_test)
# Model Performance
