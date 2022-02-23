"""
Created on Sun Feb 20 2022

@author: José Bravo
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

seed(1234)
raw_data = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/covid-geography/mmsa-icu-beds.csv')
df1 = raw_data.dropna()
print(df1.shape)
# 135 rows, 7 columns
print(list(df1.columns))
#print(df1.head())
# Alternatively
df1.sample(10)
#plt.scatter(df1.icu_beds, df1.total_at_risk)

# WIll invoke train-test-split (tts) resampling method.
def tts(data, split = 0.80):
    train = list()
    train_size = split*len(data)
    data_copy = list(data)
    while len(train) < train_size :
        index = randrange(len(data_copy))
        train.append(data_copy.pop(index))
    return train, data_copy
# Extract variables of interest.
x, y = df1.icu_beds, df1.total_at_risk
x_train, x_test = tts(x)
y_train, y_test = tts(y)
x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test)
print('Training x shape: ', x_train.shape)
print('Training y shape: ', y_train.shape)
print('Test x shape: ', x_test.shape)
print('Test y shape: ', y_test.shape)
# Build the model
model = LinearRegression()
model.fit(x_train, y_train)
y_model_train_pred = model.predict(x_train)
y_model_test_pred = model.predict(x_test)

# Model performance
# First, training mean squared error and r2
model_train_mse = mean_squared_error(y_train, y_model_train_pred)
model_train_r2 = r2_score(y_train, y_model_train_pred)

model_test_mse = mean_squared_error(y_test, y_model_test_pred)
model_test_r2 = r2_score(y_test, y_model_test_pred)
print(f'Mean Squared Error: {model_train_mse:.4f}')
print(f'r2 Value: {model_train_r2:.8f}')
# Print all the important metrics together.
model_results = pd.DataFrame(['Linear Regression', model_train_mse, model_train_r2, model_test_mse, model_test_r2]).transpose()
model_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(model_results)

########################################################################
# Following section will further analysis by trying to  determine the 
# type of distribution and remove any outliers.
#
#
#########################################################################
#import warnings
#warnings.filterwarnings('ignore')
import seaborn as sns
# plt.figure(figsize = (16, 5))
# plt.subplot(1, 2, 1)
# sns.displot(df1['total_at_risk'])
# plt.subplot(1, 2, 2)
# sns.distplot(df1['icu_beds'])
# plt.show()

response = df1['icu_beds']
#response.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
predictor = df1['total_at_risk']
#predictor.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')

# fig, axs = plt.subplots(2)
# ax1.hist(response)
# ax2.hist(predictor)
# plt.hist(response)
# plt.hist(predictor)
# plt.legend(loc = 'upper right')
# plt.show()

#fig = plt.figure()
fig = plt.figure(figsize=(18,12))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

n, bins, patches = ax1.hist(predictor, bins=20, rwidth=0.9, color='midnightblue')
ax1.set_xlabel('Total at risk')
ax1.set_ylabel('Frequency')
n, bins, patches = ax2.hist(response, bins=20, rwidth=0.9, color='yellowgreen')
ax2.set_xlabel('ICU Beds')
ax2.set_ylabel('Frequency')

# Both are skewed distributions.
#a = plt.boxplot(df1['icu_beds'])
#plt.show(a)

plt.subplot(2, 2, 3)
df1.boxplot(column = ['icu_beds'])
#boxplot1 = df1.boxplot(column = ['icu_beds'])
plt.subplot(2, 2, 4)
df1.boxplot(column = ['total_at_risk'])
#boxplot2 = df1.boxplot(column =['total_at_risk'])

# Finding the interquartile range
percentile25 =  df1['total_at_risk'].quantile(0.25)
percentile75 = df1['total_at_risk'].quantile(0.75)
iqr = percentile75 - percentile25
# Find upper and lower limit
upper_limit = percentile75 + 1.5*iqr
lower_limit = percentile25 - 1.5*iqr
# Finding outliers
df1[df1['total_at_risk'] > upper_limit]
df1[df1['total_at_risk'] < lower_limit]
# Trimming
df2 = df1[df1['total_at_risk'] < upper_limit] 
print(df2.shape)

# Comparison of plots after trimming
fig2 = plt.figure(figsize = (18, 12))
ax3 = fig2.add_subplot(2, 2, 1)
ax4 = fig2.add_subplot(2, 2, 3)
n, bins, patches = ax3.hist(df1['total_at_risk'], bins=20, rwidth=0.9, color='midnightblue')
ax3.set_xlabel('Total at risk')
ax3.set_ylabel('Frequency')
plt.subplot(2, 2, 2)
df2.boxplot(column = ['total_at_risk'])
n, bins, patches = ax4.hist(df1['icu_beds'], bins=20, rwidth=0.9, color='yellowgreen')
ax4.set_xlabel('ICU Beds')
ax4.set_ylabel('Frequency')
plt.subplot(2, 2, 4)
df2.boxplot(column = ['icu_beds'])

# Extract variables of interest.
a, b = df2.icu_beds, df2.total_at_risk
a_train, a_test = tts(a)
b_train, b_test = tts(b)
a_train, b_train = pd.DataFrame(a_train), pd.DataFrame(b_train)
a_test, b_test = pd.DataFrame(a_test), pd.DataFrame(b_test)
print('Training a shape: ', a_train.shape)
print('Training b shape: ', b_train.shape)
print('Test a shape: ', a_test.shape)
print('Test b shape: ', b_test.shape)
# Build the model
lr = LinearRegression()
lr.fit(a_train, b_train)
b_model_train_pred = model.predict(a_train)
b_model_test_pred = model.predict(a_test)

# Model performance
# First, training mean squared error and r2
lr_train_mse = mean_squared_error(b_train, b_model_train_pred)
lr_train_r2 = r2_score(b_train, b_model_train_pred)

lr_test_mse = mean_squared_error(b_test, b_model_test_pred)
lr_test_r2 = r2_score(b_test, b_model_test_pred)
print(f'Mean Squared Error: {lr_train_mse:.4f}')
print(f'r2 Value: {lr_train_r2:.8f}')
# Print all the important metrics together.
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, model_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)