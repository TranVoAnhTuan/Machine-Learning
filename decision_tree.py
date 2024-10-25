import numpy as np

# Calculate Gini Impurity
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    impurity = 1 - sum((count / len(y)) ** 2 for count in counts)
    return impurity

# Split dataset based on feature and threshold
def split_dataset(X, y, feature, threshold):
    left = np.where(X[:, feature] <= threshold)
    right = np.where(X[:, feature] > threshold)
    return X[left], y[left], X[right], y[right]

# Find the best split for a node
def best_split(X, y):
    best_gini = 1
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            _, y_left, _, y_right = split_dataset(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gini_split = (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / len(y)
            if gini_split < best_gini:
                best_gini = gini_split
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

# Build a decision tree
def build_tree(X, y, depth=0, max_depth=5):
    if len(np.unique(y)) == 1 or depth == max_depth:
        return np.argmax(np.bincount(y))

    feature, threshold = best_split(X, y)
    if feature is None:
        return np.argmax(np.bincount(y))

    X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
    left_branch = build_tree(X_left, y_left, depth + 1, max_depth)
    right_branch = build_tree(X_right, y_right, depth + 1, max_depth)

    return {"feature": feature, "threshold": threshold, "left": left_branch, "right": right_branch}

# Predict with a decision tree
def predict_tree(node, sample):
    if isinstance(node, dict):
        if sample[node["feature"]] <= node["threshold"]:
            return predict_tree(node["left"], sample)
        else:
            return predict_tree(node["right"], sample)
    else:
        return node

# Example Usage
X = np.array([[2, 3], [1, 1], [3, 2], [5, 4], [4, 5], [6, 1]])
y = np.array([0, 0, 0, 1, 1, 1])
tree = build_tree(X, y)
predictions = [predict_tree(tree, sample) for sample in X]
print("Decision Tree Predictions:", predictions)

