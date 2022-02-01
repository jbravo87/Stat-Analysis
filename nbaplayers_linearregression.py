"""
Created on Wed Jan 26 21:18:35 2022

@author: José Bravo
"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

raw_data = pd.read_csv('https://query.data.world/s/rpafmrecy2o5mkjnxa5ni22mc4kpf4')
print(raw_data.shape)
# 1340 rows, 21 columns
print(list(raw_data.columns))
print(raw_data.head())

# points and minutes
#minutes = raw_data['MIN']
minutes = np.array(raw_data['MIN'])
#minutes.reshape((-1, 1))
#print(minutes.head())

#points = raw_data['PTS']
points = np.array(raw_data['PTS'])
#print(points.head())
plt.scatter(minutes, points)
plt.show()

model = LinearRegression()
#model.fit(minutes, points)
#model.fit(raw_data['MIN'], raw_data['PTS'].shape(-1, 1))
model.fit(minutes.reshape(-1, 1), points)
#r_sq = model.score(minutes, points)
#print('Coefficient of Determination: ', r_sq)
print(f'Intercept: {model.intercept_:.3f}')
print(f'Coefficient Exposure: {model.coef_[0]:.3f}')

#y1 = zip(blocks, madethrees)
#y2 = list(y1)
#data_frame = pd.DataFrame(data = y2)

# Get the fitted values and subsequently the prediction errors.

fitted = model.predict(minutes.reshape(-1, 1))
residuals = points - fitted
print(fitted)
print(residuals)

y1 = zip(minutes, points)
y2 = list(y1)

#ax = y2.plot.scatter(x = 'Minutes', y = 'Points', figsize = (4, 4))
plot2 = plt.plot(fitted, residuals)
#for x, yactual, yfitted in zip(y2.minutes, y2.points, fitted) :
#    ax.plot(
plot2