"""
Created on Wed Jan 26 21:18:35 2022

@author: José Bravo
"""
# Import necessary libraries
from random import seed
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

seed(1234)
raw_data = pd.read_csv('https://query.data.world/s/rpafmrecy2o5mkjnxa5ni22mc4kpf4')
print(raw_data.shape)
# 1340 rows, 21 columns
print(list(raw_data.columns))
print(raw_data.head())

# To see the data type of each column in the data frame.
display(raw_data.dtypes)

# Convert TARGET_5Yrs from float to int.
raw_data['TARGET_5Yrs'] = raw_data['TARGET_5Yrs'].astype(int)

# points and minutes
# will filter the variables of interest into new data frame
df2 = raw_data[['MIN', 'PTS', 'TARGET_5Yrs']]
# Uncomment below for quick visualization.
#plt.scatter(df2.MIN, df2.PTS)

# Filter for those players that been in league longer than 5 years.
morethan5yrs = df2[df2.TARGET_5Yrs == 1]
print(morethan5yrs.shape) # 831 rows, 3 columns
plt.scatter(morethan5yrs.MIN, morethan5yrs.PTS, color = 'mediumslateblue')

# Want to invole reasampling methods.
def traintestsplit(data, split = 0.80):
    train = list()
    train_size = split*len(data)
    data_copy = list(data)
    while len(train) < train_size :
        index = randrange(len(data_copy))
        train.append(data_copy.pop(index))
    return train, data_copy

predictor = morethan5yrs.MIN
response = morethan5yrs.PTS
x_train, x_test = traintestsplit(predictor)
y_train, y_test = traintestsplit(response)
# x_train = pd.DataFrame(x_train)
# y_train = pd.DataFrame(y_train)
# x_test = pd.DataFrame(x_test)
# y_test = pd.DataFrame(y_test)
x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test)

# Now to build the model
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


# model = LinearRegression()
# #model.fit(minutes, points)
# #model.fit(raw_data['MIN'], raw_data['PTS'].shape(-1, 1))
# model.fit(minutes.reshape(-1, 1), points)
# #r_sq = model.score(minutes, points)
# #print('Coefficient of Determination: ', r_sq)
# print(f'Intercept: {model.intercept_:.3f}')
# print(f'Coefficient Exposure: {model.coef_[0]:.3f}')

# #y1 = zip(blocks, madethrees)
# #y2 = list(y1)
# #data_frame = pd.DataFrame(data = y2)

# # Get the fitted values and subsequently the prediction errors.

# fitted = model.predict(minutes.reshape(-1, 1))
# residuals = points - fitted
# print(fitted)
# print(residuals)

# y1 = zip(minutes, points)
# y2 = list(y1)

# #ax = y2.plot.scatter(x = 'Minutes', y = 'Points', figsize = (4, 4))
# plot2 = plt.plot(fitted, residuals)
# #for x, yactual, yfitted in zip(y2.minutes, y2.points, fitted) :
# #    ax.plot(
# plot2