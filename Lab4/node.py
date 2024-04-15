import copy

import numpy as np


def split_data(X, y, idx, val):
    left_mask = X[:, idx] < val
    return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])


def find_possible_splits(data):
    possible_split_points = []
    for idx in range(data.shape[0] - 1):
        if data[idx] != data[idx + 1]:
            possible_split_points.append(idx)
    return possible_split_points


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = None
        # TODO
        for idx in possible_splits:
            left_class_counts = np.bincount(y[:idx + 1], minlength=2)
            right_class_counts = np.bincount(y[idx + 1:], minlength=2)

            total_left = idx + 1
            total_right = len(y) - total_left

            gini_left = 1.0 - sum((left_class_counts / total_left) ** 2)
            gini_right = 1.0 - sum((right_class_counts / total_right) ** 2)

            gini_score = (total_left * gini_left + total_right * gini_right) / len(y)

            gain = self.gini_impurity(y) - gini_score

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        return best_idx, best_gain

    @staticmethod
    def gini_impurity(y):
        class_counts = np.bincount(y, minlength=2)
        total = len(y)
        prob = class_counts / total
        gini = 1.0 - sum(prob ** 2)
        return gini

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None

        for d in range(X.shape[1]):
            unique_values = np.unique(X[:, d])
            possible_splits = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

            for value in possible_splits:
                left_mask = X[:, d] < value
                y_left = y[left_mask]
                y_right = y[~left_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self.gini_impurity(y_left)
                gini_right = self.gini_impurity(y_right)

                weight_left = len(y_left) / len(y)
                weight_right = len(y_right) / len(y)

                impurity = weight_left * gini_left + weight_right * gini_right
                gain = self.gini_impurity(y) - impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (d, value)

        if best_split is None:
            return None, None

        return best_split[0], best_split[1]

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
