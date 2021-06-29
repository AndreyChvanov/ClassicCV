import numpy as np
from Task2.AdditionalTasks import AdaBoostC



class Cascade():
    def __init__(self, detection_rate, false_positive_rate):
        self.d = 0.995
        self.f = 0.5
        self.F_target = false_positive_rate
        self.D_target = detection_rate
        self.adaboost_list = []
        self.F = 0
        self.D = 0
    def init_weights(self, X, Y):
        face_count = len(np.where(Y == 1)[0])
        nonface_count = len(np.where(Y == 0)[0])
        W = np.where(Y == 1, 1/(2*face_count),1/(2*nonface_count))
        return W

    def fit(self, X, Y):
        F, D = 1, 1
        i = 0
        while F > self.F_target:
            adaboost = AdaBoostC.AdaBoost_C()
            current_f = 1
            print("ADABOOST = ", i)
            W = self.init_weights(X, Y)
            print("cur D", D, "cur F", F)
            # W = adaboost.add_stump(X, Y, W)
            # current_f = adaboost.false_positive_rate(X, Y, 0)
            while self.f < current_f:
                chose_threshold = 0
                W = adaboost.add_stump(X, Y, W)
                current_f = adaboost.false_positive_rate(X, Y, chose_threshold)
                pred = adaboost.predict_by_value(X, chose_threshold)
                threshold = np.linspace(0, min(pred), 1000)
                for t in threshold:
                    d = adaboost.detection_rate(X, Y, t)
                    current_f = adaboost.false_positive_rate(X, Y, t)
                    if self.d*D <= d*D:
                        print("detection rate ", d)
                        chose_threshold = t
                        print("false positive rate", current_f)
                        break
            i += 1
            F = F * adaboost.false_positive_rate(X, Y, chose_threshold)
            D = D * adaboost.detection_rate(X, Y, chose_threshold)
            self.adaboost_list.append([adaboost, chose_threshold])
            pred = adaboost.predict(X, chose_threshold)
            X = np.array([el for i, el in enumerate(X) if not (Y[i] == 0 and pred[i] ==0)])
            print(X.shape)
            Y = np.array([el for i, el in enumerate(Y) if not(Y[i] == 0 and pred[i] == 0)])
            print(Y.shape)
            print("count stump", len(adaboost.decision_stumps))
        print("F", F, "D", D)
        self.F = F
        self.D = D


def predict(cascade, data):
    pred = [predict_by_sample(cascade, sample) for sample in data]
    return np.array(pred)

def predict_by_sample(cascade, data):
    for ab in cascade.adaboost_list:
        pred = AdaBoostC.predict_by_sample(data, ab[0], ab[1])
        if pred == 0:
            return 0
    return AdaBoostC.predict_by_sample(data, cascade.adaboost_list[-1][0], cascade.adaboost_list[-1][1])

def confidence(cascade, sample):
    for ab in cascade.adaboost_list:
        pred = AdaBoostC.predict_by_sample(sample, ab[0], ab[1])
        if pred == 0:
            return 0
        else:
            output = -(1 / 2) * np.sum(np.array(ab[0].alpha)) + np.sum(
                np.array(ab[0].alpha) * np.array([stump.predict_by_sample(sample) for stump in ab[0].decision_stumps]))
            return 1 / (1 + np.exp(-output))

def detection_rate(pred, labels):
    TP = sum(pred[labels == 1] == 1)
    FN = sum(pred[labels == 1] == 0)
    return TP / (TP + FN)


def false_positive_rate(pred, labels):
    FP = sum(pred[labels == 0] == 1)
    TN = sum(pred[labels == 0] == 0)
    return FP / (FP + TN)

def get_chose_features(cascade):
    features_index = []
    for ab in cascade.adaboost_list:
        for stump in ab[0].decision_stumps:
            features_index.append(stump.best_feature)
    return features_index




                
                
                




