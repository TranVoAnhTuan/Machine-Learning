import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, n_iter=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    theta = np.zeros((n + 1, 1))

    for _ in range(n_iter):
        logits = X_b @ theta
        predictions = sigmoid(logits)
        gradient = X_b.T @ (predictions - y) / m
        theta -= learning_rate * gradient

    return theta

# Example Usage
X = 2 * np.random.rand(100, 5)
y = (4 + 3 * X[:,0] + np.random.randn(100) > 5).astype(int).reshape(-1,1)  # Creating binary labels
theta_best = logistic_regression(X, y)
# print(y)
print("Logistic Regression coefficients:", theta_best.flatten())

