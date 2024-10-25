import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classify(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x_train) for x_train in X_train]
        k_nearest = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_nearest]
        majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(majority_vote)
    return predictions

# Example Usage
X_train = np.random.rand(10, 2)
y_train = np.random.randint(0, 2, size=10)
X_test = np.random.rand(3, 2)
predictions = knn_classify(X_train, y_train, X_test, k=3)
print("KNN Predictions:", predictions)

