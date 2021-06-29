import numpy as np
from Task2.AdditionalTasks import decision_stump_C as ds



class AdaBoost_C:
    def __init__(self):
        self.decision_stumps = []
        self.alpha = []
        self.stump_err = []
    def add_stump(self, X, Y, W):
        W = W / np.sum(W)
        stump = ds.decision_stump_C(X.shape[1])
        self.stump_err.append(stump.best_stump(X, Y, W))
        print("err after add stump",self.stump_err[-1])
        self.decision_stumps.append(stump)
        correctly_samples = self.decision_stumps[-1].predict(X) == Y
        W = W * np.power(self.stump_err[-1] / (1 - self.stump_err[-1]), correctly_samples)
        self.alpha.append(np.log((1 - self.stump_err[-1]) / self.stump_err[-1]))
        return W


    def predict(self, data, threshold):
        pred = np.zeros(data.shape[0])
        pred[np.sum(np.array(self.alpha).T * np.array([stump.predict(data)
                                                          for stump in self.decision_stumps]).T, axis=1)-threshold
             >= (1 / 2) * np.sum(np.array(self.alpha))] = 1
        return pred


    def predict_by_value(self, data, threshold):
        return np.sum(np.array(self.alpha).T * np.array([stump.predict(data)
                                                  for stump in self.decision_stumps]).T, axis=1) - threshold - (1 / 2) * np.sum(np.array(self.alpha))


    def detection_rate(self, X, labels, threshold):
        pred = self.predict(X, threshold)
        TP = sum(pred[labels == 1] == 1)
        FN = sum(pred[labels == 1] == 0)
        return TP / (TP + FN)


    def false_positive_rate(self, X, labels, threshold):
        pred = self.predict(X, threshold)
        FP = sum(pred[labels == 0] == 1)
        TN = sum(pred[labels == 0] == 0)
        return FP / (FP + TN)

def predict_by_sample(data, adaboost, threshold):
    return 1 if np.sum(np.array(adaboost.alpha)*np.array([stump.predict_by_sample(data)
                                                          for stump in adaboost.decision_stumps])) - threshold \
                >= (1 / 2) * np.sum(np.array(adaboost.alpha)) else 0



