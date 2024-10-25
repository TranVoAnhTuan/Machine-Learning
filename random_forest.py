import numpy as np

# Bootstrap sampling
def bootstrap_sample(X, y):
    indices = np.random.choice(range(len(X)), size=len(X), replace=True)
    return X[indices], y[indices]

# Build a random forest with multiple decision trees
def build_forest(X, y, n_trees=5, max_depth=5):
    forest = []
    for _ in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X, y)
        tree = build_tree(X_sample, y_sample, max_depth=max_depth)
        forest.append(tree)
    return forest

# Predict with a random forest
def predict_forest(forest, sample):
    predictions = [predict_tree(tree, sample) for tree in forest]
    return np.argmax(np.bincount(predictions))

# Example Usage
forest = build_forest(X, y, n_trees=5, max_depth=3)
predictions = [predict_forest(forest, sample) for sample in X]
print("Random Forest Predictions:", predictions)

