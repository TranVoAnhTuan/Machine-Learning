import numpy as np

def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best

# Example Usage
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
theta_best = linear_regression(X, y)
print("Linear Regression coefficients:", theta_best.flatten())

