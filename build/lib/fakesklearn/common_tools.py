import numpy as np
import copy
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

class OneVsOneClassifier:
    def __init__(self, estimator = None):
        self.base_estimator_ = estimator
        self.binary = True

    def init_params(self, X, Y):
        if len(np.unique(Y)) > 2:
            self.binary = False
            self.lb = LabelBinarizer(sparse_output=False).fit(Y)
            self.Y_ = self.lb.transform(Y)

    def fit(self, X, Y):
        self.init_params(X, Y)
        if self.binary:
            return self.base_estimator_.fit(X, Y)
        self.base_model_list = [copy.deepcopy(self.base_estimator_.fit(X, Y_class)) for Y_class in self.Y_.T]
        return self

    def predict_proba(self, X):
        Y_pred_proba = np.array(np.hstack([model.predict_proba(X)[:, -1].reshape(-1,1) for model in self.base_model_list]))
        row_sum = Y_pred_proba.sum(axis = 1).reshape(-1,1)
        return Y_pred_proba/row_sum

    def predict(self, X):
        Y_pred_proba = self.predict_proba(X)
        return Y_pred_proba.argmax(axis = 1)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(Y, Y_pred)