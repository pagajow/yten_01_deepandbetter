from collections import Counter
import numpy as np

# Function to calculate entropy, a measure of uncertainty in the labels (y)
def entropy(y):
    # Create a histogram of the labels
    hist = np.bincount(y)
    # Normalize the histogram to get probabilities
    ps = hist / len(y)
    # Calculate entropy using the formula: -sum(p * log2(p)) for each probability
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

# Function to calculate information gain for a potential split
def information_gain(X_column, y, split_thresh):
    # Calculate entropy of the parent node (before the split)
    parent_entropy = entropy(y)
    
    # Split the data into left and right subsets based on the split threshold
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    
    # If one of the subsets is empty, return 0 (no information gain)
    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0
    
    # Calculate the weighted average entropy of the left and right subsets
    n = len(y)
    n_left, n_right = len(left_idxs), len(right_idxs)
    e_left, e_right = entropy(y[left_idxs]), entropy(y[right_idxs])
    child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
    
    # Information gain is the difference between the parent's entropy and the children's entropy
    ig = parent_entropy - child_entropy
    return ig

# Decision Tree Classifier class
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        # Parameters to control the depth of the tree and minimum number of samples to split
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # Fit the model by growing the tree based on input data X and target y
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    # Predict labels for new input data X
    def predict(self, X):
        # Traverse the tree for each sample in X
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    # Recursively grow the decision tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria: if max depth is reached, or all labels are the same, or the sample size is too small
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # Return a leaf node with the most common label
            leaf_value = self._most_common_label(y)
            return {"leaf": leaf_value}

        # Find the best feature and threshold for splitting
        best_feature, best_thresh = self._best_split(X, y)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

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
                # Calculate information gain for the split
                gain = information_gain(X_column, y, threshold)
                # Keep track of the split that maximizes information gain
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
        return left_idxs, right_idxs

    # Function to find the most common label in a subset of data
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    # Traverse the tree recursively to make a prediction for a single sample
    def _traverse_tree(self, x, node):
        # If it's a leaf node, return the leaf value (the predicted label)
        if "leaf" in node:
            return node["leaf"]
        # Traverse left or right based on the feature's threshold
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])


# Main function to run the classification model
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    # Load dataset (Iris)
    data = load_iris()
    X, y = data.data, data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf_custom = DecisionTreeClassifier(max_depth=10)
    clf_custom.fit(X_train, y_train)
    predictions_custom = clf_custom.predict(X_test)

    accuracy_custom = np.sum(predictions_custom == y_test) / len(y_test)
    print(f"Accuracy (custom implementation): {accuracy_custom:.4f}")

    clf_sklearn = SklearnDecisionTreeClassifier(max_depth=10, random_state=42)
    clf_sklearn.fit(X_train, y_train)
    predictions_sklearn = clf_sklearn.predict(X_test)

    accuracy_sklearn = np.sum(predictions_sklearn == y_test) / len(y_test)
    print(f"Accuracy (scikit-learn implementation): {accuracy_sklearn:.4f}")
    

