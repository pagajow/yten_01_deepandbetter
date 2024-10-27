import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else self.n_features
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return {"leaf": leaf_value}

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return {"leaf": leaf_value}

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left_subtree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {"feature": best_feature, "threshold": best_thresh, "left": left_subtree, "right": right_subtree}

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature_idx in feat_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, X_column, y, split_thresh):
        parent_entropy = self._entropy(y)

        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        ig = parent_entropy - child_entropy
        return ig

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if "leaf" in node:
            return node["leaf"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

