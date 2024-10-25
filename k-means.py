import numpy as np

def k_means(X, k, max_iter=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        new_centroids = np.array([X[closest_centroids == j].mean(axis=0) for j in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, closest_centroids

# Example Usage
X = np.random.rand(100, 2)
centroids, labels = k_means(X, k=3)
print("K-Means Centroids:", centroids)
print("Cluster assignments:", labels)

