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
plt.scatter(df1.icu_beds, df1.total_at_risk)

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




