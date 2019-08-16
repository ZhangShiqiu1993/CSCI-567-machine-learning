from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.square(y_true - y_pred))


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    y_true = np.array(real_labels)
    y_pred = np.array(predicted_labels)
    tp = np.sum(y_true * y_pred)
    if tp == 0:
        return 0.0
    precision = tp / np.sum(y_pred)
    recall = tp / np.sum(y_true)
    return 2 * (precision * recall) / (precision + recall)


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    poly_features = np.array(features)
    for i in range(2, k + 1):
        new_features = np.apply_along_axis(lambda x : np.power(x, i), 0, np.array(features))
        poly_features = np.column_stack((poly_features, new_features))
    return poly_features


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    x = np.array(point1)
    y = np.array(point2)
    return np.linalg.norm(x - y)
    

def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.dot(point1, point2)


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    d = - 0.5 * euclidean_distance(point1, point2)
    return -np.exp(d)


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        def norm(x):
            scale = np.linalg.norm(x)
            return x if scale == 0 else x / scale  
        return np.apply_along_axis(lambda x : norm(x), 1, features).tolist()


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScaler()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.scale = {}

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        def min_max(line):
            idx = line[0]
            line = line[1:]
            if idx not in self.scale:
                Min, Max = np.min(line), np.max(line)
                self.scale[idx] = (Min, Max)
            else:
                Min, Max = self.scale[idx]
            return (line - Min) / (Max - Min)
        return np.apply_along_axis(min_max, 0, np.row_stack((np.arange(len(features[0])), features))).tolist()
