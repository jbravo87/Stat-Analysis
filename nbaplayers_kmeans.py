"""
Created on Wed Jan 26 21:18:35 2022

@author: joepb
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

# Now want to focus on last column
#print(raw_data['TARGET_5Yrs'])

#print(raw_data[1:21])

# Want count of 0 or 1 (binary)
#raw_data['TARGET_5Yrs'].value_counts()

minutes = raw_data['MIN']
print(minutes.head())

points = raw_data['PTS']
print(points.head())
plt.scatter(minutes, points)
plt.show()

y1 = zip(minutes, points)
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

