import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage  

# Load your dataset

data = pd.read_csv('data.csv')

X = data.iloc[:, 1:]

# Standardize the data 
scaler = StandardScaler()
# Remove rows with NaN values
X_cleaned = X[np.logical_not(np.isnan(X).any(axis=1))]
X_scaled = scaler.fit_transform(X_cleaned)

# Define different clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=0)
hierarchical = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN(eps=0.5, min_samples=5)
gmm = GaussianMixture(n_components=3, random_state=0)

# Fit clustering algorithms to the data
kmeans_labels = kmeans.fit_predict(X_scaled)
hierarchical_labels = hierarchical.fit_predict(X_scaled)
dbscan_labels = dbscan.fit_predict(X_scaled)
gmm_labels = gmm.fit_predict(X_scaled)

# Determine optimal clusters (K-Means example)
inertia_values = []
for k in range(2, 11):  
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia_values, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters (K-Means)')
plt.show()

# Visualize clusters using PCA or t-SNE
pca = PCA(n_components=2)
tsne = TSNE(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_tsne = tsne.fit_transform(X_scaled)

# Create scatter plots for K-Means
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering (PCA)')
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering (t-SNE)')
plt.show()

# Evaluate clustering algorithms without ground truth for K-Means
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
db_index_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
ch_score_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)

# Print the evaluation metrics for K-Means
print(f'Silhouette Score for K-Means: {silhouette_kmeans}')
print(f'Davies-Bouldin Index for K-Means: {db_index_kmeans}')
print(f'Calinski-Harabasz Score for K-Means: {ch_score_kmeans}')

# Visualize hierarchical clustering as a dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = linkage(X_scaled, method='ward')  # You may adjust the method as needed
    plt.figure(figsize=(8, 5))
    dendrogram(linkage_matrix, **kwargs)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

# Plot the hierarchical dendrogram for Hierarchical Clustering
plot_dendrogram(hierarchical, labels=hierarchical_labels)

# Evaluate clustering algorithms without ground truth for Agglomerative Clustering
silhouette_agglomerative = silhouette_score(X_scaled, hierarchical_labels)
db_index_agglomerative = davies_bouldin_score(X_scaled, hierarchical_labels)
ch_score_agglomerative = calinski_harabasz_score(X_scaled, hierarchical_labels)

# Print the evaluation metrics for Agglomerative Clustering
print(f'Silhouette Score for Agglomerative Clustering: {silhouette_agglomerative}')
print(f'Davies-Bouldin Index for Agglomerative Clustering: {db_index_agglomerative}')
print(f'Calinski-Harabasz Score for Agglomerative Clustering: {ch_score_agglomerative}')

# Visualize clusters for DBSCAN using PCA or t-SNE
X_pca_dbscan = pca.fit_transform(X_scaled)
X_tsne_dbscan = tsne.fit_transform(X_scaled)

# Create scatter plots for DBSCAN
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca_dbscan[:, 0], X_pca_dbscan[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering (PCA)')
plt.subplot(1, 2, 2)
plt.scatter(X_tsne_dbscan[:, 0], X_tsne_dbscan[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering (t-SNE)')
plt.show()
dbscan = DBSCAN(eps=0.2, min_samples=9)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate DBSCAN using silhouette score and Davies-Bouldin Index
#silhouette_dbscan = silhouette_score(X_scaled, dbscan_labels)
#db_index_dbscan = davies_bouldin_score(X_scaled, dbscan_labels)

# Print the evaluation metrics for DBSCAN
#print(f'Silhouette Score for DBSCAN: {silhouette_dbscan}')
#print(f'Davies-Bouldin Index for DBSCAN: {db_index_dbscan}')

# Gaussian Mixture Models (GMM)
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_labels = gmm.fit_predict(X_scaled)

# Visualize clusters for GMM using PCA or t-SNE
X_pca_gmm = pca.fit_transform(X_scaled)
X_tsne_gmm = tsne.fit_transform(X_scaled)

# Create scatter plots for GMM
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca_gmm[:, 0], X_pca_gmm[:, 1], c=gmm_labels, cmap='viridis')
plt.title('Gaussian Mixture Models Clustering (PCA)')
plt.subplot(1, 2, 2)
plt.scatter(X_tsne_gmm[:, 0], X_tsne_gmm[:, 1], c=gmm_labels, cmap='viridis')
plt.title('Gaussian Mixture Models Clustering (t-SNE)')
plt.show()

# GMM doesn't have Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
# Therefore use AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) for GMM evaluation.
aic = gmm.aic(X_scaled)
bic = gmm.bic(X_scaled)

# Print AIC and BIC for GMM
print(f'AIC for GMM: {aic}')
print(f'BIC for GMM: {bic}')
