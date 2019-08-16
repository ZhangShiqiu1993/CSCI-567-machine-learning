from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        x = numpy.mat(numpy.column_stack((numpy.ones(len(features)), features)))
        y = numpy.mat(values).T
        if numpy.linalg.matrix_rank(x) == x.shape[1]:
            self.weights = ((numpy.linalg.pinv(x.T.dot(x))).dot(x.T)).dot(y)
        else:
            self.weights = numpy.linalg.lstsq(x, y, rcond=-1)[0]
        

    def predict(self, features: List[List[float]]) -> List[float]:
        x = numpy.mat(numpy.column_stack((numpy.ones(len(features)), features)))
        return x.dot(self.weights).T.tolist()[0]

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.T.tolist()[0]


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        x = numpy.mat(numpy.column_stack((numpy.ones(len(features)), features)))
        y = numpy.mat(values).T
        self.weights = (((x.T.dot(x) + self.alpha * numpy.eye(x.shape[1]))).I.dot(x.T)).dot(y)
        
    def predict(self, features: List[List[float]]) -> List[float]:
        x = numpy.mat(numpy.column_stack((numpy.ones(len(features)), features)))
        return x.dot(self.weights).T.tolist()[0]

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.T.tolist()[0]


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
