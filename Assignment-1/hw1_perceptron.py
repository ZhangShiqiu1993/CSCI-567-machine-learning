from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        seq = [x for x in range(len(features))]
        threshold = self.margin / 2
        converge = False
        scale = np.linalg.norm(features)
        for iteration in range(self.max_iteration):
            if converge:
                break
            converge = True
            np.random.shuffle(seq)
            for i in seq:
                pred = np.dot(self.w, features[i])
                y = 0
                if pred > threshold:
                    y = 1
                elif pred < -threshold:
                    y = -1
                if y != labels[i]:
                    self.w = np.add(self.w, np.dot(labels[i], features[i]))
                    converge = False
        self.w = self.w.tolist()
        return converge
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        return np.apply_along_axis(lambda x : 1 if np.dot(self.w, x) > 0 else -1, 1, features)

    def get_weights(self) -> List[float]:
        return self.w
