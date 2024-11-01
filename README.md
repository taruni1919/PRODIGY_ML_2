# Step 1: Data Preparation

We'll simulate customer purchase data first.

python

import pandas as pd import numpy as np

#Simulating customer purchase history data np.random.seed(42)

n_customers = 100

data = pd.DataFrame({

'CustomerID': range(1, n_customers + 1),

'TotalSpend': np.random.rand(n_customers) 1000, #Total spend between 0 and 1000

'Frequency': np.random.randint(1, 20, size=n_customers), # Purchases per month

'Avg TransactionValue': np.random.rand(n_customers) 100, #Average transaction value

2

'DaysSinceLast Purchase': np.random.randint(1, 365, size=n_customers) # Days since last purchase

})

print("Sample Data:")

print(data.head())

# Step 2: Implement K-means Clustering

inertia.append(kmeans.inertia_) Now we'll run the K-means algorithm.

python

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#Selecting features

features = data[['TotalSpend', 'Frequency', 'AvgTransactionValue', 'DaysSinceLastPurchase']]

#Normalizing the features

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

inertia = []

# Determine optimal K using the Elbow method

k_values = range(1, 11)

for k in k_values: kmeans = KMeans(n_clusters=k, random_state=42)
#Plotting the elbow graph

plt.figure(figsize=(10, 6))

plt.plot(k_values, inertia, marker='o')

plt.title('Elbow Method for Optimal K')

plt.xlabel('Number of clusters (K)')

plt.ylabel('Inertia")

plt.grid()

plt.show()

4 After examining the elbow plot, let's say we choose *K = *.

# Step 3: Run K-means with Chosen K

python

#Running K-means with the chosen K

optimal_k = 4

kmeans = KMeans(n_clusters optimal_k, random_state=42) clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to original data

data['Cluster'] = clusters

# Display the mean values for each cluster

cluster_summary = data.groupby('Cluster').mean()

print("\nCluster Summary:")

print(cluster_summary)
