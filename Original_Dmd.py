
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load dataset
abalone = fetch_ucirepo(id=1)
df = abalone.data.features
df['Age'] = abalone.data.targets  # Add target column

# Display basic information
print(df.info())
print(df.describe())

# Convert categorical 'Sex' to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Convert any boolean values to integers (if any exist)
df_encoded = df_encoded.astype(float)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Outlier Detection Using IQR
Q1 = df_encoded.quantile(0.25)
Q3 = df_encoded.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_encoded < (Q1 - 1.5 * IQR)) | (df_encoded > (Q3 + 1.5 * IQR))).astype(int).sum()

print("Number of outliers per feature:\n", outliers)

# Choose optimal k using the elbow method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Apply K-Means (choosing k=3 based on elbow method)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_encoded['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Apply DBSCAN
dbscan = DBSCAN(eps=1, min_samples=5)
df_encoded['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Evaluate Clustering Performance
silhouette_kmeans = silhouette_score(X_scaled, df_encoded['KMeans_Cluster'])
db_index_kmeans = davies_bouldin_score(X_scaled, df_encoded['KMeans_Cluster'])

print(f"K-Means Silhouette Score: {silhouette_kmeans:.3f}")
print(f"K-Means Davies-Bouldin Index: {db_index_kmeans:.3f}")

# Evaluate DBSCAN (excluding noise points)
dbscan_labels = df_encoded['DBSCAN_Cluster']
non_noise_mask = dbscan_labels != -1
X_dbscan_filtered = X_scaled[non_noise_mask]
labels_dbscan_filtered = dbscan_labels[non_noise_mask]

if len(np.unique(labels_dbscan_filtered)) > 1:
    silhouette_dbscan = silhouette_score(X_dbscan_filtered, labels_dbscan_filtered)
    db_index_dbscan = davies_bouldin_score(X_dbscan_filtered, labels_dbscan_filtered)
    print(f"DBSCAN Silhouette Score: {silhouette_dbscan:.3f}")
    print(f"DBSCAN Davies-Bouldin Index: {db_index_dbscan:.3f}")
else:
    print("DBSCAN produced less than 2 clusters (excluding noise); evaluation not possible.")

# Apply PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_encoded['KMeans_Cluster'], cmap='viridis', edgecolor='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering Visualization')
plt.colorbar()
plt.show()

# DBSCAN Visualization
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_encoded['DBSCAN_Cluster'], cmap='plasma', edgecolor='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Density-Based Clustering Visualization')
plt.colorbar()
plt.show()
