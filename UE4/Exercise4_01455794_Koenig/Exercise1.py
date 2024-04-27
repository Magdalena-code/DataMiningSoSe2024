import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import folium

data = pd.read_csv('whisky.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#print(data.columns)

#print(data.head())

#data cleaning
numerical_columns = ['Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco', 'Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral']

# Create a boolean mask for non-numerical values
non_numerical_mask = data[numerical_columns].applymap(lambda x: not isinstance(x, (int, float)))

# Print non-numerical values
for col in numerical_columns:
    non_numerical_values = data[non_numerical_mask[col]][col]
    if not non_numerical_values.empty:
        print(f"Non-numerical values in column '{col}':")
        print(non_numerical_values)

for col in numerical_columns:
    # Convert all entries to numeric, setting errors='coerce' will convert non-convertible values to NaN
    data[col] = pd.to_numeric(data[col], errors='coerce')

num_distillery_unique = data['Distillery'].nunique()
total_rows = len(data)

print(f"Number of unique distilleries: {num_distillery_unique}, Total rows: {total_rows}")

duplicate_mask = data.duplicated(subset=['Distillery'], keep=False)  # Mark all duplicates

# Display which professions are not unique
non_unique_distillery = data[duplicate_mask]['Distillery'].unique()
print("Non-unique distilleries:", non_unique_distillery)

# Remove all rows that are duplicates in the 'profession' column
# keep='first' to keep the first occurrence, use keep=False to remove all duplicates
data = data.drop_duplicates(subset=['Distillery'], keep='first')

# Check the result
print("Number of rows after removal:", len(data))

missing_values_count = data.isnull().sum()
missing_percentage = (missing_values_count / len(data)) * 100
print("Missing Percentage")
print(missing_percentage)

# Create a boxplot for all numerical columns
plt.figure(figsize=(10, 6))  # Adjust the size of the figure as needed
data[numerical_columns].boxplot()
plt.title('Boxplot of all numerical columns')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()
#No visiable outliners or noise

# Identify non-numerical columns
non_numerical_columns = data.select_dtypes(include=['object']).columns

# Create a bar chart for each non-numerical column
for col in non_numerical_columns:
    plt.figure(figsize=(10, 6))  # Adjust the size of the figure as needed
    data[col].value_counts().plot(kind='bar')
    plt.title(f'Bar chart of {col}')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.show()

#Clustering

# not needed for clustering
data_for_clustering = data.drop(['RowID', 'Distillery', 'Postcode', 'Latitude', 'Longitude'], axis=1)

# Hierarchical Clustering
# Generate the linkage matrix using Ward's method
linkage_matrix = linkage(data_for_clustering, 'ward')

# Plot the dendrogram
plt.figure(figsize=(30, 14))
dendrogram(linkage_matrix, labels=data['Distillery'].values, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Distilleries')
plt.ylabel('Distance')
plt.show()

# K-Means Clustering
# Determining the optimal number of clusters using silhouette score
silhouette_scores = []

 # Testing for 2 to 9 clusters
list_k = list(range(2, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    cluster_labels = km.fit_predict(data_for_clustering)
    silhouette_avg = silhouette_score(data_for_clustering, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Silhouette Score for {k} clusters: {silhouette_avg:.4f}")

# Plot SSE for each *k*
plt.figure(figsize=(6, 6))
plt.plot(list_k, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()

#Optimal Number of Clusters: Based on this plot, the optimal number of clusters according to the silhouette score is 2, as this is where the score is highest, suggesting the data points are, on average, well matched to
#their own cluster and poorly matched to neighboring clusters.
#Stability of Clustering: The significant drop after 2 clusters could suggest that any further subdivision of the data does not capture well-defined clusters and may start to split clusters that should remain together.
#Cluster Quality: The silhouette score for 2 clusters isn't particularly close to 1, which indicates that while 2 clusters are the best configuration among the ones tested, there is room for improvement in the clustering quality.
#This might be due to the data itself (e.g., it does not naturally cluster well, or there is noise), or it could be a limitation of the hierarchical clustering method for this particular dataset.

#2 clusters as the silhouette score is highest at 2

# Cutting the dendrogram to get two clusters
cluster_labels_2 = fcluster(linkage_matrix, 2, criterion='maxclust')
cluster_labels_5 = fcluster(linkage_matrix, 5, criterion='maxclust')

# Apply PCA to reduce to two dimensions for visualization
pca_2 = PCA(n_components=2)
reduced_data_2 = pca_2.fit_transform(data_for_clustering)


# Scatter plot of the two principal components colored by cluster label
plt.figure(figsize=(10, 8))
plt.scatter(reduced_data_2[cluster_labels_2==1, 0], reduced_data_2[cluster_labels_2==1, 1], c='blue', label='Cluster 1')
plt.scatter(reduced_data_2[cluster_labels_2==2, 0], reduced_data_2[cluster_labels_2==2, 1], c='red', label='Cluster 2')
plt.title('Hierarchical Clustering with 2 Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Scatter plot of the five principal components colored by cluster label
# Scatter plot of the first two principal components colored by cluster label
plt.figure(figsize=(10, 8))
for i, color, lbl in zip(range(1, 6), ['blue', 'red', 'green', 'orange', 'pink'], ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']):
    plt.scatter(reduced_data_2[cluster_labels_5 == i, 0], reduced_data_2[cluster_labels_5 == i, 1], c=color, label=lbl)
plt.title('Hierarchical Clustering with 5 Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#PC1: This is the direction in the multidimensional space of your data that accounts for the largest amount of variance from the original dataset. It represents the greatest spread of data points along a single dimension and captures the most significant pattern within the data.
#PC2: This is the direction that captures the second-largest amount of variance and is orthogonal (at a right angle) to PC1. It represents the next most significant pattern in the data that is independent of PC1.

# Print the contributions of features to PC1 in order
print("Contribution of features to PC1:")
contributions_pc1 = sorted(zip(data_for_clustering.columns, pca_2.components_[0]), key=lambda x: x[1], reverse=True)
for name, loading in contributions_pc1:
    print(f"{name}: {loading:.2f}")

# Print the contributions of features to PC2 in order
print("\nContribution of features to PC2:")
contributions_pc2 = sorted(zip(data_for_clustering.columns, pca_2.components_[1]), key=lambda x: x[1], reverse=True)
for name, loading in contributions_pc2:
    print(f"{name}: {loading:.2f}")


kmeans_2 = KMeans(n_clusters=2)
kmeans_5 = KMeans(n_clusters=5)
clusters_2 = kmeans_2.fit_predict(data_for_clustering)
clusters_5 = kmeans_5.fit_predict(data_for_clustering)
data_for_clustering['Cluster_2'] = clusters_2
data_for_clustering['Cluster_5'] = clusters_5

data['Cluster_2'] = clusters_2
data['Cluster_5'] = clusters_5

print(data_for_clustering.head())

# Check distribution of Cluster
print(data_for_clustering['Cluster_2'].value_counts())
print(data_for_clustering['Cluster_5'].value_counts())

# Calculate the mean of each column per cluster for 2 clusters
mean_values_cluster_2 = data_for_clustering.groupby('Cluster_2').mean()

# Print the mean values
print("Mean values per cluster for 2 clusters:")
print(mean_values_cluster_2)

# Calculate the mean of each column per cluster for 5 clusters
mean_values_cluster_5 = data_for_clustering.groupby('Cluster_5').mean()

# Print the mean values
print("\nMean values per cluster for 5 clusters:")
print(mean_values_cluster_5)


# Function to create a radar plot for each cluster
#Help of chatgpt
def create_radar_chart(data, labels, title):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color='red', alpha=0.25)
    ax.plot(angles, data, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(title, size=15, color='red', y=1.1)

# Calculate mean values for each cluster and create radar plots
for i in range(2):
    cluster_data = data_for_clustering[data_for_clustering['Cluster_2'] == i].drop(['Cluster_2', 'Cluster_5'], axis=1)
    mean_values = cluster_data.mean().values
    create_radar_chart(mean_values, cluster_data.columns, f'Radar plots for 2 clusters: Cluster {i+1} Flavor Profile')

for i in range(5):
    cluster_data = data_for_clustering[data_for_clustering['Cluster_5'] == i].drop(['Cluster_2', 'Cluster_5'], axis=1)
    mean_values = cluster_data.mean().values
    create_radar_chart(mean_values, cluster_data.columns, f'Radar plots for 5 clusters: Cluster {i+1} Flavor Profile')
plt.show()

#Help of chatgpt
map_2 = folium.Map(location=[56.4907, -4.2026], zoom_start=6)
for _, row in data.iterrows():
    folium.CircleMarker(
        location=(row['Longitude'], row['Latitude']),
        radius=5,
        color='red' if row['Cluster_2'] == 0 else 'blue',
        fill=True,
        fill_color='red' if row['Cluster_2'] == 0 else 'blue',
        popup=f"{row['Distillery']} (Cluster_2 {row['Cluster_2']})"
    ).add_to(map_2)


#Help of chatgpt
map_5 = folium.Map(location=[56.4907, -4.2026], zoom_start=6)

# Define colors for the five clusters
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Add a circle marker for each distillery
for _, row in data.iterrows():
    # Use the cluster number to choose the color from the colors list
    cluster_color = colors[row['Cluster_5']]

    folium.CircleMarker(
        location=(row['Longitude'], row['Latitude']),
        radius=5,
        color=cluster_color,
        fill=True,
        fill_color=cluster_color,
        popup=f"{row['Distillery']} (Cluster_5 {row['Cluster_5']})"
    ).add_to(map_5)

# Show the maps
map_2.save('map_2.html')
map_5.save('map_5.html')

webbrowser.open('map_2.html')
webbrowser.open('map_5.html')