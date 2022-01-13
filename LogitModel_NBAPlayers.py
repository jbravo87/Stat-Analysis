# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:04:19 2022

@author: joepb
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "white")
sns.set(style = "whitegrid", color_codes = True)

raw_data = pd.read_csv('https://query.data.world/s/rpafmrecy2o5mkjnxa5ni22mc4kpf4')
print(raw_data.shape)
# 1340 rows, 21 columns
print(list(raw_data.columns))
print(raw_data.head())

# Now want to focus on last column
print(raw_data['TARGET_5Yrs'])

#print(raw_data[1:21])

# Want count of 0 or 1 (binary)
raw_data['TARGET_5Yrs'].value_counts()

