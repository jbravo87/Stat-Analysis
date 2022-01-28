"""
Created on Wed Jan 26 21:18:35 2022

@author: José Bravo
"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
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
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(data_frame)
    Sum_of_squared_distances.append(kmeans.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('Value of K')
plt.ylabel('Sum of squared distances/Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters :
    # Initialize kmeans
    kmeans = KMeans( n_clusters = num_clusters )
    kmeans.fit( data_frame )
    cluster_labels = kmeans.labels_
    
    # Silhouette Score
    silhouette_avg.append(silhouette_score(data_frame, cluster_labels))

plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()