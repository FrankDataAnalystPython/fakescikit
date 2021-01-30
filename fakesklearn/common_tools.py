import numpy as np
import copy
import itertools
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, recall_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

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


class fakeSearchCV:
    def __init__(self,
                 estimator,
                 param_grid,
                 scoring='accuracy',
                 cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv

    def init_params(self):
        self.all_p = [{k: v for k, v in zip(p, i)}
                      for i in itertools.product(*self.param_grid.values())]
        self.base_estimators_ = [clone(self.estimator.set_params(**i))
                                 for i in self.all_p]

    def fit(self, X, Y):
        self.init_params()
        self.Results = []
        for model in self.base_estimators_:
            cv_score = cross_val_score(model, X, Y, cv=self.cv, scoring=self.scoring).mean()
            self.Results.append(cv_score)
        self.final_df_result = np.array(self.Results)
        self.best_score_ = self.final_df_result.max()
        self.best_params_ = self.all_p[self.final_df_result.argmax()]
        self.best_estimator_ = self.estimator.set_params(**self.best_params_).fit(X, Y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, Y):
        return sklearn.metrics.SCORERS[self.scoring](self.best_estimator_, X, Y)