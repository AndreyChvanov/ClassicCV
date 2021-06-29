import numpy as np


class decision_stump:
    def __init__(self,NoF,):
        self.number_of_features = NoF
        self.best_threshold = 0
        self.best_polar = 0
        self.best_feature = 0

    def _fit(self, f, x_train, y_train, weights):
        X = x_train[:, f]
        min_err = weights.sum()
        p_best = 1
        best_threshold = X[0]
        for i, threshold in enumerate(np.linspace(min(X), max(X), 10)):
            for p in [-1, 1]:
                cur_result = np.zeros_like(y_train)
                cur_result[X * p < p * threshold] = 1
                error = np.sum(np.abs(cur_result - y_train) * weights)
                if error < min_err:
                    min_err = error
                    p_best = p
                    best_threshold = threshold
        return min_err, best_threshold, p_best

    def best_stump(self, X, Y, W):
        best_f = 0
        p_best = 1
        best_threshold = 0
        min_err = W.sum()
        for f in range(self.number_of_features):
            error, threshold, p = self._fit(f, X, Y, W)
            if error < min_err:
                min_err = error
                best_f = f
                best_threshold = threshold
                p_best = p
        self.best_threshold = best_threshold
        self.best_feature = best_f
        self.best_polar = p_best
        return min_err

    def predict(self, data):
        X = data[:, self.best_feature]
        predict = np.zeros(data.shape[0])
        predict[X * self.best_polar < self.best_polar * self.best_threshold] = 1
        return predict

    def predict_by_sample(self,sample):
        return 1 if sample[self.best_feature]*self.best_polar < self.best_polar*self.best_threshold else 0
