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
from scipy import stats
from sklearn.utils import resample

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
# Convert just columns "a" and "b"
#sample_df.iloc[:, 1] = sample_df.iloc[:, 1].apply(pd.to_numeric)
#sample_df.iloc[:, 2] = sample_df.iloc[:, 2].apply(pd.to_numeric)
sample_df.loc[:, 'orbitperiod'] = sample_df.loc[:, 'orbitperiod'].apply(pd.to_numeric)
sample_df.loc[:, 'eccentricity'] = sample_df.loc[:, 'eccentricity'].apply(pd.to_numeric)

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
plot1 = sns.displot(df4.orbitperiod, color = 'green')
plt.title("title")
plt.show(plot1)
plot2 = sns.displot(df4.eccentricity, color = 'darkorange')
plt.title("title")
plt.show(plot2)
plot3 = plt.hist(df4.orbitperiod, color = "olivedrab", density = True, edgecolor = 'black', bins=20)
plt.title("Histrogram of Orbit Period column")
plt.show(plot3)
# Plot4 options changes the x-y dependence
plot4 = plt.scatter([df4.eccentricity],[df4.orbitperiod], color = 'darkcyan')
#plot4 = plt.scatter([df4.orbitperiod], [df4.eccentricity])
plt.title("Scatterplot")
plt.show(plot4)

# Now logic for the bootstrapping method.
# First, the orbit period column .
orbitperiod = df4.iloc[:, 1]
results1 = []
for nrepeat in range(1000):
    sample = resample(orbitperiod)
    results1.append(sample.median())
results1 = pd.Series(results1)
print('\nBootstrap Statistics for orbit period column: \n')
print(f'original: {orbitperiod.median()}')
print(f'bias: {results1.mean() - orbitperiod.median()}')
print(f'std. error: {results1.std()}')

print("\nOrbit period mean: %.4f" % orbitperiod.mean())
#np.random.seed(seed = 3)
np.random.seed(seed = 1554)
# Create a sample of 20 orbit period data
sample20 = resample(orbitperiod, n_samples = 20, replace = False)
print('Sample of 20 mean: %.4f' % sample20.mean())
results3 = []
for nrepeat in range(500) :
    sample = resample(sample20)
    results3.append(sample.mean())
results3 = pd.Series(results3)

# Confidence Interval
confidence_interval = list(results3.quantile([0.05, 0.95]))
ax = results3.plot.hist(bins=30, figsize=(4, 3), color='coral')
ax.plot(confidence_interval, [55, 55], color='black')
for x in confidence_interval:
    ax.plot([x, x], [0, 65], color='black')
    ax.text(x, 70, f'{x:.0f}', 
            horizontalalignment='center', verticalalignment='center')
ax.text(sum(confidence_interval) / 2, 60, '90% interval',
        horizontalalignment='center', verticalalignment='center')

mean_orbitperiod = results3.mean()
ax.plot([mean_orbitperiod, mean_orbitperiod], [0, 50], color='black', linestyle='--')
ax.text(mean_orbitperiod, 10, f'Mean: {mean_orbitperiod:.0f}',
        bbox=dict(facecolor='white', edgecolor='white', alpha=0.5),
        horizontalalignment='center', verticalalignment='center')
ax.set_ylim(0, 80)
ax.set_ylabel('Counts')

plt.tight_layout()
plt.title('Confidence Interval for Orbit Period')
plt.show()
# Plot above is the bootstrap confidence interal for the orbit period based on a sample of 20.
# Want to visualize the distribution and compare to normal distribution using QQ plot
fig, ax1 = plt.subplots(figsize = (4, 4))
norm_sample = stats.norm.rvs(size = 100)
#stats.probplot(norm_sample, plot = ax1)
#stats.probplot(orbitperiod, dist = stats.expon, plot = ax1)
lam = 2.13
#stats.probplot(norm_sample, dist = stats.tukeylambda(lam), plot = ax1)
stats.probplot(orbitperiod, dist = stats.tukeylambda(lam), plot = ax1)
#stats.probplot(orbitperiod, dist = stats.t(lam), plot = ax1)
plt.tight_layout()
plt.show()

# Check out the column names
#print(df4.columns)

# Now want to create a Random Forest model.
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import utils

#predictors = ['planetname', 'eccentricity']
predictors = ['eccentricity']
response = ['orbitperiod']

X = df4[predictors]
y = df4[response]
le = preprocessing.LabelEncoder()
encoded = le.fit_transform(y)

rf = RandomForestClassifier(n_estimators = 500, random_state = 1, oob_score = True)
#rf.fit(predictors, response)
rf.fit(X, encoded.ravel())
print(rf.oob_decision_function_)