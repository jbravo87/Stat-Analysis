"""
Created on Mon Mar  7 19:13:28 2022

First attempt at a bootstrap approach to determine the distribution of planetary data.

@author: José Bravo
"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from scipy.stats import chi2_contingency
#from scipy.stats import chi2
#from fitter import Fitter, get_common_distributions, get_distributions

raw_data = pd.read_csv('J:\\datasets\\PS_2022.02.27_18.46.12.csv')
# Need to remove the first thirteen rows.
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
sample_df['orbitperiod'] = sample_df['orbitperiod'].astype(float)
sample_df['eccentricity'] = sample_df['eccentricity'].astype(float)
#sample_df['orbitperiod'].astype(float)
#sample_df['eccentricity'].astype(float)
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

# Distributional plots
plot1 = sns.displot(df4.orbitperiod)
plt.title("title")
plt.show(plot1)
plot2 = sns.displot(df4.eccentricity)
plt.title("title")
plt.show(plot2)
plot3 = plt.hist(df4.orbitperiod, density = True, edgecolor = 'black', bins=20)
plt.title("Histrogram of Orbit Period column")
plt.show(plot3)
# Plot4 options changes the x-y dependence
plot4 = plt.scatter([df4.eccentricity],[df4.orbitperiod])
#plot4 = plt.scatter([df4.orbitperiod], [df4.eccentricity])
plt.title("Scatterplot")
plt.show(plot4)

#lessthan12yrs = 