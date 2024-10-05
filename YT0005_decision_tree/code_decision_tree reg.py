import numpy as np

# Function to compute variance of the target values (y)
def variance(y):
    return np.var(y)

# Function to compute variance reduction after splitting the data
def variance_reduction(X_column, y, split_thresh):
    # Calculate the variance of the parent node
    parent_variance = variance(y)
    
    # Split data into left and right subsets based on the split threshold
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()

    # If one of the subsets is empty, return 0 (no variance reduction)
    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0

    # Calculate the weighted variance of the left and right subsets
    n = len(y)
    n_left, n_right = len(left_idxs), len(right_idxs)
    var_left, var_right = variance(y[left_idxs]), variance(y[right_idxs])

    # Weighted average of the variances of the two subsets
    weighted_variance = (n_left / n) * var_left + (n_right / n) * var_right
    
    # Variance reduction is the difference between parent and child variances
    var_gain = parent_variance - weighted_variance
    return var_gain

# Decision Tree Regressor class
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        # Parameters to control the depth of the tree and minimum number of samples to split
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # Fit the model by growing the tree based on input data X and target y
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    # Predict values for new input data X
    def predict(self, X):
        # Traverse the tree for each sample in X
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    # Recursively grow the decision tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Stopping criteria: if max depth is reached or the sample size is too small
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            # Return a leaf node with the mean value of the target
            leaf_value = np.mean(y)
            return {"leaf": leaf_value}

        # Find the best feature and threshold for splitting
        best_feature, best_thresh = self._best_split(X, y)

        # If no valid split is found, return a leaf node with the mean value
        if best_feature is None or best_thresh is None:
            leaf_value = np.mean(y)
            return {"leaf": leaf_value}

        # Split the data based on the best feature and threshold
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        # If one of the splits is empty, return a leaf node
        if left_idxs is None or right_idxs is None:
            leaf_value = np.mean(y)
            return {"leaf": leaf_value}

        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        # Return the current node with the feature, threshold, and left/right subtrees
        return {"feature": best_feature, "threshold": best_thresh, "left": left_subtree, "right": right_subtree}

    # Find the best feature and threshold to split on
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        n_features = X.shape[1]

        # Iterate through each feature to find the best split
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)  # Unique thresholds for splitting
            for threshold in thresholds:
                # Calculate variance reduction for the split
                gain = variance_reduction(X_column, y, threshold)
                # Keep track of the split that maximizes variance reduction
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold

        # Return the best feature and threshold for splitting
        return split_idx, split_thresh

    # Split the data into left and right subsets based on a threshold
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        # If one of the subsets is empty, return None
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return None, None

        return left_idxs, right_idxs

    # Traverse the tree recursively to make a prediction for a single sample
    def _traverse_tree(self, x, node):
        # If it's a leaf node, return the leaf value
        if "leaf" in node:
            return node["leaf"]
        # Traverse left or right based on the feature's threshold
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

# Main function to run the regression model
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_california_housing

    # Load dataset (California Housing)
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor_custom = DecisionTreeRegressor(max_depth=10)
    regressor_custom.fit(X_train, y_train)
    predictions_custom = regressor_custom.predict(X_test)

    mse_custom = np.mean((predictions_custom - y_test) ** 2)
    mean_actual = np.mean(y_test)
    relative_mse_custom = mse_custom / mean_actual
    print(f"MSE (custom implementation): {mse_custom:.4f}")
    print(f"Relative MSE (custom implementation): {relative_mse_custom:.4f}")

    regressor_sklearn = SklearnDecisionTreeRegressor(max_depth=10, random_state=42)
    regressor_sklearn.fit(X_train, y_train)
    predictions_sklearn = regressor_sklearn.predict(X_test)

    mse_sklearn = np.mean((predictions_sklearn - y_test) ** 2)
    relative_mse_sklearn = mse_sklearn / mean_actual
    print(f"MSE (scikit-learn implementation): {mse_sklearn:.4f}")
    print(f"Relative MSE (scikit-learn implementation): {relative_mse_sklearn:.4f}")
