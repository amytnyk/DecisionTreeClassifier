import operator
import numpy as np
import scipy.stats
from dataclasses import dataclass


@dataclass
class Node:
    feature_index: int = 0
    threshold: float = 0
    left = None
    right = None
    value: float = None


class InnerNode(Node):
    def __init__(self, feature_index, threshold, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


class LeafNode(Node):
    def __init__(self, value):
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth

    def gini(self, data) -> float:
        groups = list(zip(*np.unique(data[:, -1], return_counts=True)))
        return 1 - sum(map(lambda x: (x[1] / data.shape[0]) ** 2, groups))

    def split_data(self, data) -> tuple[int, int, np.array, np.array]:
        possible_splits = []
        for feature in range(data.shape[1] - 1):
            for threshold in np.unique(data[:, feature]):
                lower_set = data[data[:, feature] <= threshold, :]
                higher_set = data[data[:, feature] > threshold, :]
                weight_lower = lower_set.shape[0] / data.shape[0]
                weight_higher = higher_set.shape[0] / data.shape[0]
                ig = self.gini(data) - weight_lower * self.gini(lower_set) - weight_higher * self.gini(higher_set)
                possible_splits.append((ig, (feature, threshold, lower_set, higher_set)))

        return max(possible_splits, key=operator.itemgetter(0))[1]

    def build_tree(self, data, depth=0) -> Node:
        if depth == self.max_depth or np.all(data[:, -1] == data[:, -1][0]):
            return LeafNode(scipy.stats.mode(data[:, -1])[0][0])

        feature, threshold, lower_set, higher_set = self.split_data(data)
        if len(lower_set) == 0 or len(higher_set) == 0:
            return LeafNode(scipy.stats.mode(data[:, -1])[0][0])

        left = self.build_tree(lower_set, depth + 1)
        right = self.build_tree(higher_set, depth + 1)

        return InnerNode(feature, threshold, left, right)

    def fit(self, X, y):
        self.root = self.build_tree(np.column_stack((X, y)))
        return self

    def make_prediction(self, x: np.array, node: Node) -> float:
        if isinstance(node, LeafNode):
            return node.value
        else:
            return self.make_prediction(x, node.left if x[node.feature_index] <= node.threshold else node.right)

    def predict(self, X_test):
        return np.array(list(map(lambda x: self.make_prediction(x, self.root), X_test)))
