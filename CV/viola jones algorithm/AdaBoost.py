import numpy as np
from Task2 import decision_stump as ds
import os
import pickle


class AdaBoost:
    def __init__(self, number_of_weak_classifiers, weights):
        self.NWC = number_of_weak_classifiers
        self.alfa = np.zeros(self.NWC)
        self.W = weights
        self.stumps = []

    def fit(self, X, Y):
        stump_error = []
        for t in range(self.NWC):
            print(t)
            self.W = self.W / np.sum(self.W)
            stump = ds.decision_stump(X.shape[1])
            error = stump.best_stump(X, Y, self.W)
            self.stumps.append(stump)
            stump_error.append(error)
            correctly_samples = stump.predict(X) == Y
            self.W = self.W * np.power(error / (1 - error), correctly_samples)
            self.alfa[t] = np.log((1 - error) / error)
        return stump_error


def confidence(adaboost, X):
    output = -(1/2)*np.sum(np.array(adaboost.alfa))+np.sum(np.array(adaboost.alfa)*np.array([stump.predict_by_sample(X) for stump in adaboost.stumps]))
    return 1 / (1 + np.exp(-output))


def predict(data, adaboost):
    pred = np.zeros(data.shape[0])
    pred[np.sum(np.array(adaboost.alfa).T * np.array([stump.predict(data)
                                                      for stump in adaboost.stumps]).T, axis=1)
         >= (1 / 2) * np.sum(np.array(adaboost.alfa))] = 1
    return pred
def predict_by_sample(data, adaboost):
    return 1 if np.sum(np.array(adaboost.alfa)*np.array([stump.predict_by_sample(data) for stump in adaboost.stumps])) >= (1 / 2) * np.sum(np.array(adaboost.alfa)) else 0

def accuracy(ab_predict, labels):
    return (ab_predict == labels).mean()

def true_positive_rate(ab_predict, labels):
    TP = sum(ab_predict[labels == 1] == 1)
    FN = sum(ab_predict[labels == 1] == 0)
    return TP/(TP+FN)

def false_positive_rate(ab_predict, labels):
    FP = sum(ab_predict[labels == 0] == 1)
    TN = sum(ab_predict[labels == 0] == 0)
    return FP/(FP+TN)

def save_model(adaboost, filename):
    f = open(filename, 'wb')
    pickle.dump(adaboost, f)
    f.close()

def load_model(filename):
    f = open(filename, 'rb')
    adaboost = pickle.load(f)
    f.close()
    return adaboost


