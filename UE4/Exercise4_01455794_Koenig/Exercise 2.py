#done togethere with Thomas Hollin

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import webbrowser
import numpy as np
import folium
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



#import Data
# Read csv
uber = pd.read_csv('uber.csv', sep=',',encoding='utf-8')

print(uber.info())
print(uber.describe())

#data cleaning
missing_values_count = uber.isnull().sum()
missing_percentage = (missing_values_count / len(uber)) * 100
print("Missing Percentage")
print(missing_percentage)

# Drop rows with any missing values
uber = uber.dropna()

print("New length of DataFrame after dropping rows with missing values:", len(uber))

ubercl = uber.iloc[:, 1:3]

print(ubercl)


#data visualization
plt.scatter(uber['Lat'], uber['Lon'])
plt.title('Distribution coordinates')
plt.show()


#K-Means - 3 clusters
k_clusters = 3

kmeans = KMeans(n_clusters=k_clusters, n_init=3)

#compute cluster centers and predict index
clusters = kmeans.fit_predict(ubercl)

#Inertia measures how well a dataset was clustered by K-Means.
#It is calculated by measuring the distance between each data
#point and its centroid, squaring this distance, and summing
#these squares across one cluster. A good model is one with
#low inertia AND a low number of clusters ( K ). - https://www.geeksforgeeks.org/python-mean-squared-error/

optimal_sse = kmeans.inertia_

print(f"Sum of Squared Errors for k={k_clusters}: {optimal_sse}")

# Display cluster centers

centers = kmeans.cluster_centers_
print("K-Means Cluster Centers (Longitude, Latitude):")
print(centers)


# Add cluster labels back to the original DataFrame
uber['Cluster'] = clusters


plt.figure(figsize=(10, 8))
plt.scatter(uber['Lon'], uber['Lat'], c=uber['Cluster'], cmap='viridis', marker='o', alpha=0.5)
plt.title('Uber Pickup Points Clustered')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')
plt.show()

#K-Means - 8 clusters
k_clusters = 8

kmeans = KMeans(n_clusters=k_clusters, n_init=3)

#compute cluster centers and predict index
clusters = kmeans.fit_predict(ubercl)

optimal_sse = kmeans.inertia_

print(f"Sum of Squared Errors for k={k_clusters}: {optimal_sse}")

# Display cluster centers
centers = kmeans.cluster_centers_
print("K-Means Cluster Centers (Longitude, Latitude):")
print(centers)

# Add cluster labels back to the original DataFrame
uber['Cluster'] = clusters



plt.figure(figsize=(10, 8))
plt.scatter(uber['Lon'], uber['Lat'], c=uber['Cluster'], cmap='viridis', marker='o', alpha=0.5)
plt.title('Uber Pickup Points Clustered')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')
plt.show()


#k-Means - 12 clusters

k_clusters = 12

kmeans = KMeans(n_clusters=k_clusters, n_init=3)

#compute cluster centers and predict index
clusters = kmeans.fit_predict(ubercl)

optimal_sse = kmeans.inertia_

print(f"Sum of Squared Errors for k={k_clusters}: {optimal_sse}")

# Display cluster centers
centers = kmeans.cluster_centers_
print("K-Means Cluster Centers (Longitude, Latitude):")
print(centers)

# Add cluster labels back to the original DataFrame
uber['Cluster'] = clusters

plt.figure(figsize=(10, 8))
plt.scatter(uber['Lon'], uber['Lat'], c=uber['Cluster'], cmap='viridis', marker='o', alpha=0.5)
plt.title('Uber Pickup Points Clustered')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')
plt.show()



#Model based clustering - Gaussian Mixture Model 3 clusters
#Flexibility: Clusters can have different sizes, different shapes, and different orientations in the dataset, which provides flexibility in fitting the data.
#Soft Clustering: Each data point has a probability of belonging to each (vs. Kmeans where every point belongs to exactly one value)
# of the clusters rather than being hard-assigned, allowing for more nuanced interpretations.


# Standardizing the data
scaler = StandardScaler()
uber_scaled = scaler.fit_transform(ubercl)

# Initialize and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(uber_scaled)
hub_centers = gmm.means_
clusters = gmm.fit_predict(uber_scaled)

# Add cluster labels to the original DataFrame
uber['Cluster_GMM'] = clusters

#converts the cluster centers, which are calculated in the scaled dataset's space back to the original data space
hub_centers_gmm = scaler.inverse_transform(hub_centers)

# Print the lower bound and the number of iterations - the higher the lower bound, the better fitting the model
# and the lower to number of interactions the the more efficient.
print(f"Lower bound on the log-likelihood for g={gmm.n_components}: {gmm.lower_bound_}")
print(f"Number of iterations for g={gmm.n_components}: {gmm.n_iter_}")

# Print the hub locations
print("GMM Cluster Centers (Latitude, Longitude):")
print(hub_centers_gmm)

# Plotting the clusters
plt.figure(figsize=(10, 8))
plt.scatter(uber['Lon'], uber['Lat'], c=uber['Cluster_GMM'], cmap='viridis', marker='o', alpha=0.5)
plt.title('Uber Pickup Points Clustered (GMM)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')
plt.show()


#Model based clustering - Gaussian Mixture Model 8 clusters

# Standardizing the data
scaler = StandardScaler()
uber_scaled = scaler.fit_transform(ubercl)

# Initialize and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=8, random_state=42)
gmm.fit(uber_scaled)
hub_centers = gmm.means_
clusters = gmm.fit_predict(uber_scaled)

# Add cluster labels to the original DataFrame
uber['Cluster_GMM'] = clusters

#converts the cluster centers, which are calculated in the scaled dataset's space back to the original data space
hub_centers_gmm = scaler.inverse_transform(hub_centers)

# Print the lower bound and the number of iterations - the higher the lower bound, the better fitting the model
# and the lower to number of interactions the the more efficient.
print(f"Lower bound on the log-likelihood for g={gmm.n_components}: {gmm.lower_bound_}")
print(f"Number of iterations for g={gmm.n_components}: {gmm.n_iter_}")

# Print the hub locations
print("GMM Cluster Centers (Latitude, Longitude):")
print(hub_centers_gmm)

# Plotting the clusters
plt.figure(figsize=(10, 8))
plt.scatter(uber['Lon'], uber['Lat'], c=uber['Cluster_GMM'], cmap='viridis', marker='o', alpha=0.5)
plt.title('Uber Pickup Points Clustered (GMM)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')
plt.show()

#Model based clustering - Gaussian Mixture Model 12 clusters

# Standardizing the data
scaler = StandardScaler()
uber_scaled = scaler.fit_transform(ubercl)

# Initialize and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=12, random_state=42)
gmm.fit(uber_scaled)
hub_centers = gmm.means_
clusters = gmm.fit_predict(uber_scaled)

# Add cluster labels to the original DataFrame
uber['Cluster_GMM'] = clusters

#converts the cluster centers, which are calculated in the scaled dataset's space back to the original data space
hub_centers_gmm = scaler.inverse_transform(hub_centers)

# Print the lower bound and the number of iterations - the higher the lower bound, the better fitting the model
# and the lower to number of interactions the the more efficient.
print(f"Lower bound on the log-likelihood for g={gmm.n_components}: {gmm.lower_bound_}")
print(f"Number of iterations for g={gmm.n_components}: {gmm.n_iter_}")

# Print the hub locations
print("GMM Cluster Centers (Latitude, Longitude):")
print(hub_centers_gmm)

# Plotting the clusters
plt.figure(figsize=(10, 8))
plt.scatter(uber['Lon'], uber['Lat'], c=uber['Cluster_GMM'], cmap='viridis', marker='o', alpha=0.5)
plt.title('Uber Pickup Points Clustered (GMM)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')
plt.show()



#Kmeans vs. GMM:
#Real-world Shape Adaptability: GMM is particularly good for real-world data that naturally forms elliptical clusters rather
# than spherical ones, which is often the case in geospatial data like the Uber dataset.
#Uncertainty in Cluster Assignments: GMM provides probabilities for cluster assignments, which is useful when you are unsure
# about the boundaries between clusters or when data points could logically belong to multiple clusters.

#Visualizing the clusters for 12 centers (k-means)

map_kmean = folium.Map(location=[40.70, -74.00], zoom_start=9)

#add every clustercenter
for center in centers:
    folium.Marker(
        [center[0], center[1]],
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(map_kmean)
map_kmean.save('uber_hubs_map_kmean.html')



#Visualizing the clusters for 12 centers (GMM)
map_gmm = folium.Map(location=[40.70, -74.00], zoom_start=9)

for hub_centers_gmm in hub_centers_gmm:
    folium.Marker(
        [hub_centers_gmm[0], hub_centers_gmm[1]],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(map_gmm)
map_gmm.save('uber_hubs_map_gmm.html')

webbrowser.open('uber_hubs_map_gmm.html')
webbrowser.open('uber_hubs_map_kmean.html')


#conclusion:
#Given the likely geographic variability and the potential overlap in areas of high Uber activity, GMM seems more
# fitting for this case. It provides a richer, more detailed understanding of how pickup points cluster together.
#if speed and simplicity are crucial, and the data distribution assumptions of K-means (spherical clusters, similar sizes)
# hold reasonably true in preliminary analysis, K-means might still be a viable option.



