from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.features = numpy.array(features)
        self.labels = numpy.array(labels)

    def predict(self, features: List[List[float]]) -> List[int]:
        def _predict(x, loo):
            distance = numpy.apply_along_axis(lambda y : self.distance_function(x, y), 1, self.features)

            if loo:
                knn = numpy.argpartition(distance, self.k + 1)[1: self.k + 1]
            else:
                knn = numpy.argpartition(distance, self.k)[:self.k]

            return 1 if numpy.mean(self.labels[knn]) > 0.5 else 0
            
        X_test = numpy.array(features)
        loo = numpy.array_equal(X_test, self.features)
        y_pred = numpy.apply_along_axis(lambda x : _predict(x, loo), 1, X_test)
        return y_pred
        


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
