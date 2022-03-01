# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:04:43 2022

@author: joepb
"""
# Import necessary libraries
from random import seed
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# raw_data = pd.read_csv('J:\\datasets\\stellardata.csv')
# plt.scatter(raw_data.kepmag, raw_data.logg) #quick scatteplot
raw_data = pd.read_csv('J:\\datasets\\PS_2022.02.27_18.46.12.csv')
# Need to remove the first thirteen rows.
raw_data = raw_data.drop(raw_data.index[range(14)])
print(raw_data.columns)
# first data frame with columns of interest
df1 = raw_data.iloc[:, [0, 1, 5]]
df1.reset_index(drop=True, inplace=True)
#df1.reset_index()
df1.rename(columns={'# This file was produced by the NASA Exoplanet Archive  http://exoplanetarchive.ipac.caltech.edu':'planetname', 'Unnamed: 1':'orbitperiod', 'Unnamed: 5':'eccentricity'}, inplace=True)
df1 = df1.drop(df1.index[range(1)])
# Next data frame will ne one with NA values removed
df2 = df1.dropna()
df2.reset_index(drop=True, inplace=True)
# plt.scatter(df2.orbitperiod, df2.eccentricity)
# sample_df = df2.iloc[0:9]
sample_df = df2.iloc[0:789]
#######
# Notice that the data type for the last two columns 
# is string and not double or numeric
#######
print(type(sample_df.iloc[0][1]))
print(type(sample_df.iloc[0][2]))
sample_df['orbitperiod'] = sample_df['orbitperiod'].astype(float)
sample_df['eccentricity'] = sample_df['eccentricity'].astype(float)
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
y1 = sample_df.groupby('planetname', as_index = False)['orbitperiod'].mean()
y2 = sample_df.groupby('planetname', as_index = False)['eccentricity'].mean()
# final_df = pd.concat([y1['orbitperiod'], y2['eccentricity']])
result = y1.merge(y2, on=['planetname'])
import seaborn as sns
sns.distplot(result.orbitperiod)
sns.distplot(result.eccentricity)
