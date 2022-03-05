# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:04:43 2022

@author: joepb
"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from fitter import Fitter, get_common_distributions, get_distributions
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

from scipy import stats
def check_normality(data):
    test_stat_normality, p_value_normality=stats.shapiro(data)
    print("p value:%.4f" % p_value_normality)
    if p_value_normality <0.05:
        print("Reject null hypothesis >> The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis >> The data is normally distributed")

check_normality(result.eccentricity)
result.boxplot(column = ['orbitperiod'])
result.boxplot(column = ['eccentricity'])

# # Finding the interquartile range for the orbit period
# percentile25 =  result['orbitperiod'].quantile(0.25)
# percentile75 = result['orbitperiod'].quantile(0.75)
# iqr = percentile75 - percentile25
# # Find upper and lower limit
# upper_limit = percentile75 + 1.5*iqr
# lower_limit = percentile25 - 1.5*iqr
# # Finding outliers
# result[result['orbitperiod'] > upper_limit]
# result[result['orbitperiod'] < lower_limit]
# # Trimming
# result = result[result['orbitperiod'] < upper_limit] 

# Turn the above logic into a function
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

print(result.shape)

# To get the number of rows/observations
result.shape[0]

# Finding the interquartile range for the eccentricity
# percent25 =  result['eccentricity'].quantile(0.25)
# percent75 = result['eccentricity'].quantile(0.75)
# iqr2 = percent25 - percent75
# # Find upper and lower limit
# upper_limit2 = percent75 + 1.5*iqr2
# lower_limit2 = percent25 - 1.5*iqr2
# # Finding outliers
# result[result['eccentricity'] > upper_limit2]
# result[result['eccentricity'] < lower_limit2]
# Trimming
#result = result[result['eccentricity'] < upper_limit] 

# Following to determine the distirbution
get_common_distributions()
x = iqr(result, 'eccentricity')
x1 = iqr(x, 'orbitperiod')
plt.hist(x1.orbitperiod, density = True, edgecolor = 'black', bins=20)
sns.distplot(x1.eccentricity)
x5 = Fitter(x1.orbitperiod, distributions = get_distributions())
x5.fit()
x5.summary()
x5.get_best(method = 'sumsquare_error')

obs = np.array([[result.orbitperiod],[result.eccentricity]])
chi2, p, dof, ex = chi2_contingency(obs, correction=False)
print("expected frequencies:\n ", np.round(ex,2))
print("degrees of freedom:", dof)
print("test stat :%.4f" % chi2)
print("p value:%.4f" % p)
# from scipy.stats import chi2
# ## calculate critical stat

# alpha = 0.01
# df = (5-1)*(2-1)
# critical_stat = chi2.ppf((1-alpha), df)
# print("critical stat:%.4f" % critical_stat)