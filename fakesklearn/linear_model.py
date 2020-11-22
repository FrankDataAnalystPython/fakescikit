import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegression(BaseEstimator):
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept

    def init_params(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()

    def add_ones(self, X):
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack((ones, X))
        return X

    def fit(self, X, Y):
        self.init_params(X, Y)
        tmp_X = self.add_ones(X)

        tmp_X = np.mat(tmp_X)
        tmp_Y = np.mat(self.Y).T

        XTX = tmp_X.T * tmp_X
        if np.linalg.det(XTX) == 0:
            raise ZeroDivisionError

        self.W = XTX.I * tmp_X.T * tmp_Y
        if self.fit_intercept:
            self.coef_ = np.ravel(self.W)[1:]
            self.intercept_ = np.ravel(self.W)[0]
        else:
            self.coef_ = self.W
        return self

    def predict(self, X):
        tmp_X = self.add_ones(X)
        tmp_X = np.mat(tmp_X)
        Y_pred = tmp_X * self.W
        return np.ravel(Y_pred)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred)

class Ridge(BaseEstimator):
    def __init__(self,
                 learning_rate = 0.01, max_iter = 500, tol = 1e-4,
                 solver = 'Batch', fit_intercept = True, alpha = 1
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def init_params(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
        self.std = StandardScaler().fit(self.X)
        self.X = self.std.transform(self.X)
        self.W = np.mat(np.zeros((self.X.shape[1] + 1, 1)))
        self.GD_dict = {'Batch' : self._cal_grad_Batch,
                        'Stocastic' : self._cal_grad_Stocastic,
                        'Mini' : self._cal_grad_Mini}

    def add_ones(self, X):
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack((ones, X))
        return X

    def _cal_grad_Batch(self, X, Y):
        return 2 / X.shape[0] * (X.T * (X * self.W - Y) + self.alpha * self.W)

    def _cal_grad_Stocastic(self, X, Y):
        j = np.random.choice(range(X.shape[0]), 1)
        X_j, Y_j = X[j], Y[j]
        return 2 / X.shape[0] * (X_j.T * (X_j * self.W - Y_j) + self.alpha * self.W)

    def _cal_grad_Mini(self, X, Y):
        j = np.random.choice(range(X.shape[0]), 32)
        X_j, Y_j = X[j], Y[j]
        return 2 / X.shape[0] * (X_j.T * (X_j * self.W - Y_j) + self.alpha * self.W)

    def fit(self, X, Y):
        self.init_params(X, Y)
        tmp_X = np.mat(self.add_ones(self.X))
        tmp_Y = np.mat(self.Y).T
        steps = 0

        while steps < self.max_iter:
            steps += 1
            gd = self.GD_dict[self.solver](tmp_X, tmp_Y)
            if gd.T * gd < self.tol:
                break
            self.W -= self.learning_rate * gd

        if self.fit_intercept:
            self.coef_ = np.ravel(self.W)[1:]
            self.intercept_ = np.ravel(self.W)[0]
        else:
            self.coef_ = self.W
        return self

    def predict(self, X):
        X = self.std.transform(X)
        tmp_X = self.add_ones(X)
        tmp_X = np.mat(tmp_X)
        Y_pred = tmp_X * self.W
        return np.ravel(Y_pred)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred)

class LogisticRegression(BaseEstimator):
    def __init__(self,
                 tol = 1e-4, C = 1, fit_intercept = True, max_iter = 5000,
                 learning_rate = 0.1, solver = 'sag'
                 ):
        self.tol = tol
        self.C = 1/C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.solver = solver

    def init_params(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
        self.std = StandardScaler().fit(self.X)
        self.X = self.std.transform(self.X)
        self.W = np.mat(np.zeros((self.X.shape[1] + 1, 1)))
        self.delta_dict = {'sag' : self.SAG,
                           'newton' : self.Newton}

    def add_ones(self, X):
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack((ones, X))
        return X

    def sigmod(self, X, w):
        z = X * w
        return 1/(1 + np.exp(-z))

    def cal_sigmoid_error(self, X, Y):
        sigmod_result = self.sigmod(X, self.W)
        error = sigmod_result - Y
        return error, sigmod_result

    def SAG(self, X, Y):
        if self.solver == 'sag':
            j = np.random.choice(range(X.shape[0]), 32)
            X_j, Y_j = X[j], Y[j]
        else:
            X_j, Y_j = X, Y
        error, sigmod_result = self.cal_sigmoid_error(X_j, Y_j)
        gd = 1/X.shape[0] * (X_j.T * error + self.C * self.W)
        if self.solver == 'sag':
            return gd
        elif self.solver == 'newton':
            return gd, sigmod_result

    def Hessian(self, X, sigmod_result):
        sigmod_result = np.ravel(sigmod_result)
        B = np.diag(sigmod_result)
        return 1/X.shape[0] * X.T * B * X + self.C * np.eye(X.shape[1])

    def Newton(self, X, Y):
        gd, sigmod_result = self.SAG(X, Y)
        H = self.Hessian(X, sigmod_result)
        return H.I * gd

    def fit(self, X, Y):
        self.init_params(X, Y)
        tmp_X = np.mat(self.add_ones(self.X))
        tmp_Y = np.mat(self.Y).T
        steps = 0

        while steps < self.max_iter:
            steps += 1
            delta = self.delta_dict[self.solver](tmp_X, tmp_Y)
            while delta.T * delta < self.tol:
                break
            if self.solver == 'sag':
                self.W -= self.learning_rate * delta
            elif self.solver == 'newton':
                self.W -= delta

        if self.fit_intercept:
            self.coef_ = np.ravel(self.W)[1:]
            self.intercept_ = np.ravel(self.W)[0]
        else:
            self.coef_ = self.W
        self.iterations = steps
        return self

    def predict_proba(self, X):
        X = self.std.transform(X)
        tmp_X = np.mat(self.add_ones(X))
        Y_proba_1 = self.sigmod(tmp_X, self.W)
        Y_proba_0 = 1 - Y_proba_1
        Y_proba = np.hstack((Y_proba_0, Y_proba_1))
        return Y_proba

    def predict(self, X):
        Y_pred_proba = self.predict_proba(X)
        return Y_pred_proba.argmax(axis = 1)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(Y, Y_pred)

if __name__ == '__main__':
    boston = load_boston()
    X = boston['data']
    Y = boston['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state = 42)

    LR = LinearRegression().fit(Xtrain, Ytrain)
    print(LR.score(Xtrain, Ytrain), LR.score(Xtest, Ytest))

    ridge = Ridge(max_iter = 5000, solver = 'Mini', learning_rate = 0.1, alpha = 0.5).fit(Xtrain, Ytrain)
    print(ridge.score(Xtrain, Ytrain), ridge.score(Xtest, Ytest))

    bc = load_breast_cancer()
    # bc = load_digits(n_class=2)
    X = bc['data']
    Y = bc['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=46)

    LogR = LogisticRegression(C = 1, solver = 'newton', max_iter = 5000).fit(Xtrain, Ytrain)
    print(LogR.score(Xtrain, Ytrain), LogR.score(Xtest, Ytest), LogR.iterations)

    print('done')