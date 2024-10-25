import numpy as np

def pca(X, n_components=2):
    X_meaned = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    reduced_eigenvectors = sorted_eigenvectors[:, :n_components]
    X_reduced = X_meaned @ reduced_eigenvectors
    return X_reduced

# Example Usage
X = np.random.rand(100, 5)
X_reduced = pca(X, n_components=2)
print("Reduced Data:", X_reduced[:5])  # Showing first 5 points in reduced space

