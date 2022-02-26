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

raw_data = pd.read_csv('J:\\datasets\\stellardata.csv')
plt.scatter(raw_data.kepmag, raw_data.logg) #quick scatteplot