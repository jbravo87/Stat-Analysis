"""
Created on Wed Jan 26 21:18:35 2022

@author: José Bravo
"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

raw_data = pd.read_csv('https://query.data.world/s/rpafmrecy2o5mkjnxa5ni22mc4kpf4')
print(raw_data.shape)
# 1340 rows, 21 columns
print(list(raw_data.columns))
print(raw_data.head())

# points and assists
#assists = raw_data['AST']
#points = raw_data['PTS']

#plt.scatter(assists, points)
#plt.show()

#y1 = zip(minutes, points)
#y2 = list(y1)
#data_frame = pd.DataFrame(data = y2)

# rebounds = raw_data['REB']
# blocks = raw_data['BLK']
# plt.scatter(rebounds, blocks)
# plt.show()

# madethrees = raw_data['3P Made']
# rebounds = raw_data['REB']
# plt.scatter(rebounds, madethrees)
# plt.show()

blocks = raw_data['BLK']
madethrees = raw_data['3P Made']
plt.scatter(blocks, madethrees)
plt.show()

y1 = zip(blocks, madethrees)
y2 = list(y1)
data_frame = pd.DataFrame(data = y2)


Sum_of_squared_distances = []
K = range(1, 10)

for num_clusters in K :
    kmeans1 = KMeans(n_clusters = num_clusters)
    kmeans1.fit(data_frame)
    Sum_of_squared_distances.append(kmeans1.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('Value of K')
plt.ylabel('Sum of squared distances/Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters :
    # Initialize kmeans
    kmeans2 = KMeans( n_clusters = num_clusters )
    kmeans2.fit( data_frame )
    cluster_labels = kmeans2.labels_
    
    # Silhouette Score
    silhouette_avg.append(silhouette_score(data_frame, cluster_labels))

plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()

# Load data
##pca = PCA(2)

# Transform data
##data_frame = pca.fit_transform(data_frame)
##data_frame.shape
##print(data_frame.shape)

#list(data_frame.columns)
df_t = data_frame.T
print(df_t)

# According to silhouette analysis need no more than 3 clusters.
kmeans = KMeans(n_clusters = 3)
label = kmeans.fit_predict(data_frame)
filtered_label0 = data_frame[label == 0]
##label = kmeans.fit_predict(df_t)
##filtered_label0 = data_frame[label == 0]


# Plot results
#plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
# plt.scatter(filtered_label0.loc[:, 0], filtered_label0.loc[:, 1])
# plt.xlim(-0.1, 4.0)
# plt.ylim(-0.20, 2.2)
# plt.show()


# Filter rows of original data
filtered_label1 = data_frame[label == 1]
filtered_label2 = data_frame[label == 2]

# Plotting the results
plt.scatter(filtered_label0.loc[:, 0], filtered_label0.loc[:, 1], color = 'limegreen')
plt.scatter(filtered_label1.loc[:, 0], filtered_label1.loc[:, 1], color = 'deepskyblue')
plt.scatter(filtered_label2.loc[:, 0], filtered_label2.loc[:, 1], color = 'black')
plt.xlabel('Average Blocks')
plt.ylabel('Average Made 3PTS') 
plt.title('Blocks vs. Made 3PT')
plt.show()