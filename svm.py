import numpy as np

def svm_sgd(X, y, learning_rate=0.01, lambda_param=0.01, n_iter=1000):
    m, n = X.shape
    y = np.where(y <= 0, -1, 1)  # Convert to -1, 1 labels
    weights = np.zeros(n)
    bias = 0

    for _ in range(n_iter):
        for i in range(m):
            if y[i] * (np.dot(X[i], weights) - bias) >= 1:
                weights -= learning_rate * (2 * lambda_param * weights)
            else:
                weights -= learning_rate * (2 * lambda_param * weights - np.dot(X[i], y[i]))
                bias -= learning_rate * y[i]
    
    return weights, bias

# Example Usage
X = np.random.randn(10, 2)
y = np.random.choice([-1, 1], 10)  # Random binary labels for illustration
weights, bias = svm_sgd(X, y)
print("SVM weights:", weights)
print("SVM bias:", bias)

