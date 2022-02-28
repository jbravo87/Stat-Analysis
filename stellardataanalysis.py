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
plt.scatter(df2.orbitperiod, df2.eccentricity)