import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T
        
        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return
    
    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return
    
    def predict(self, features: List[List[float]]) -> List[int]:
        ########################################################
        # implement "predict"
        ########################################################
        h = np.zeros(len(features), dtype=np.float64)
        for t in range(self.T):
            h += np.array(self.clfs_picked[t].predict(features)) * self.betas[t]
        ans = np.where(h >= 0, 1, -1)
        return ans.tolist()


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return
    
    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        #  implement "train"
        ############################################################
        N = len(features)
        w = np.ones(N) / N
        labels = np.array(labels)
        
        for t in range(self.T):
            epsilon_t = 1
            h_t = None
            for classifier in self.clfs:
                h_x = classifier.predict(features)
                h_x = np.array(h_x)
                epsilon = np.sum(w[h_x != labels])
                if epsilon < epsilon_t:
                    epsilon_t = epsilon
                    h_t = classifier
            self.clfs_picked.append(h_t)
            
            beta_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            self.betas.append(beta_t)
            w *= np.exp(-beta_t * (labels * np.array(h_t.predict(features))))
            w /= np.sum(w)
    
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)


class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return
    
    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # implement "train"
        ############################################################
        N = len(features)
        pie = np.ones(N) / 2
        labels = np.array(labels)
        f_t = np.zeros(N)
        for t in range(self.T):
            w = pie * (1 - pie)
            z_t = ((labels + 1) / 2 - pie) / w
            
            epsilon_t = N * 10e3
            h_t = None
            for classifier in self.clfs:
                h_x = classifier.predict(features)
                h_x = np.array(h_x)
                epsilon = np.dot(w, np.square(z_t - h_x))
                if epsilon < epsilon_t:
                    epsilon_t = epsilon
                    h_t = classifier
            self.clfs_picked.append(h_t)
            self.betas.append(0.5)
            h_x = np.array(h_t.predict(features))
            f_t += 0.5 * h_x
            pie = 1 / (1 + np.exp(-2 * f_t))
    
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
