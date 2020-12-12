import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
import copy

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

class _LogisticRegression(BaseEstimator):
    def __init__(self,
                 tol = 1e-4, C = 1, fit_intercept = True, max_iter = 5000,
                 learning_rate = 0.1, solver = 'sag'
                 ):
        super().__init__()
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
        if self.fit_intercept:
            self.W = np.mat(np.zeros((self.X.shape[1] + 1, 1)))
        else:
            self.W = np.mat(np.zeros((self.X.shape[1], 1)))
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


class LogisticRegression(BaseEstimator):
    def __init__(self,
                 tol = 1e-4, C = 1, fit_intercept = True, max_iter = 5000,
                 learning_rate = 0.1, solver = 'sag'
                 ):
        super().__init__()
        self.tol = tol
        self.C = 1/C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.solver = solver
        self.binary = True


    def init_params(self, X, Y):
        if len(np.unique(Y)) > 2:
            self.binary = False
            if len(Y.shape) == 1:
                self.onehot = OneHotEncoder(sparse = False).fit(Y.reshape(-1, 1))
                self.Y_ = self.onehot.transform(Y.reshape(-1, 1))

        estimator_params = ("tol", "C", "fit_intercept",
                            "max_iter", "learning_rate",
                            "solver")
        self.base_estimator_ = _LogisticRegression(**{p: getattr(self, p) for p in estimator_params})


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

class _Perceptron(BaseEstimator):
    def __init__(self, gaussian_kernel=True,
                 normalize = True,
                 max_iter = 500,
                 gamma = 1e-2,
                 predict_probability=True,
                 Log_model = 'sklearn'
                 ):
        super().__init__()
        self.gaussian_kernel = gaussian_kernel
        self.normalize = normalize
        self.max_iter = max_iter
        self.gamma = gamma
        self.predict_probability = predict_probability
        self.Log_model = Log_model

    def init_params(self, X, Y):
        self.N, self.M = X.shape
        self.X = X.copy()
        self.Y = Y.copy()
        self.Y[self.Y == 0] = -1

        if self.normalize:
            self.stand = StandardScaler().fit(self.X)
            self.X = self.stand.transform(self.X)

        self.a = np.zeros(self.N)
        self.Gram_matrix = self.Kernel(self.X, self.X)


    def _gaussian_dot(self, X_i, X_j):
        return np.exp(-self.gamma * ((X_i - X_j) ** 2).sum())

    def Kernel(self, X1, X2):
        N = X1.shape[0]
        M = X2.shape[0]
        if self.gaussian_kernel:
            Gram_matrix = np.zeros((N, M))
            for i in range(N):
                X_i = X1[i, :]
                for j in range(M):
                    X_j = X2[j, :]
                    Gram_matrix[i, j] = self._gaussian_dot(X_i, X_j)
        else:
            Gram_matrix = X1 @ X2.T
        return Gram_matrix

    def fit(self, X, Y):
        self.init_params(X, Y)
        steps = 0
        while steps < self.max_iter:
            steps += 1
            missing_index = -1
            for j in range(self.X.shape[0]):
                checking = self.Y[j] * ((self.a * self.Y * self.Gram_matrix[j, :]).sum() + self.a @ self.Y)
                if checking <= 0:
                    missing_index = j
                    break
            if missing_index == -1:
                break
            self.a[missing_index] += 1

        if self.predict_probability:
            X_log = self.decision_function(self.X).reshape(-1, 1)
            self.std = StandardScaler().fit(X_log)
            X_log = self.std.transform(X_log)
            if self.Log_model == 'sklearn':
                self.logistic_model = sklearn_LogisticRegression(solver='lbfgs').fit(X_log, self.Y)
            elif self.Log_model == 'fakesklearn':
                self.logistic_model = LogisticRegression(solver='newton').fit(X_log, self.Y)
        self.steps = steps
        return self

    def _predict_dis(self, col):
        return (self.a * self.Y * col).sum() + self.a @ self.Y

    def decision_function(self, X):
        pred_Gram_matrix = pd.DataFrame(self.Kernel(self.X, X))
        X_log = pred_Gram_matrix.apply(self._predict_dis, axis=0).values
        return X_log

    def predict_proba(self, X):
        if self.normalize:
            X = self.stand.transform(X)
        X_log = self.decision_function(X).reshape(-1, 1)
        X_log = self.std.transform(X_log)
        return self.logistic_model.predict_proba(X_log)

    def predict(self, X):
        if self.predict_probability:
            Y_pred_proba = self.predict_proba(X)
            return Y_pred_proba.argmax(axis = 1)

        Y_pred = np.sign(self.decision_function(X))
        Y_pred[Y_pred == -1] = 0
        return Y_pred

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(Y, Y_pred)

class Perceptron(BaseEstimator):
    def __init__(self,
                 gaussian_kernel=True,
                 normalize=True,
                 max_iter=500,
                 gamma=1e-2,
                 predict_probability=True,
                 Log_model='sklearn'
                 ):
        super().__init__()
        self.gaussian_kernel = gaussian_kernel
        self.normalize = normalize
        self.max_iter = max_iter
        self.gamma = gamma
        self.predict_probability = predict_probability
        self.Log_model = Log_model
        self.binary = True

    def init_params(self, X, Y):
        if len(np.unique(Y)) > 2:
            self.predict_probability = True
            self.binary = False
            if len(Y.shape) == 1:
                self.onehot = OneHotEncoder(sparse = False).fit(Y.reshape(-1, 1))
                self.Y_ = self.onehot.transform(Y.reshape(-1, 1))

        estimator_params = ("gaussian_kernel", "normalize", "max_iter",
                            "gamma", "predict_probability",
                            "Log_model")
        self.base_estimator_ = _Perceptron(**{p: getattr(self, p) for p in estimator_params})

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


if __name__ == '__main__':
    boston = load_boston()
    X = boston['data']
    Y = boston['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state = 43)

    LR = LinearRegression().fit(Xtrain, Ytrain)
    print(LR.score(Xtrain, Ytrain), LR.score(Xtest, Ytest))

    ridge = Ridge(max_iter = 5000, solver = 'Mini', learning_rate = 0.1, alpha = 0.5).fit(Xtrain, Ytrain)
    print(ridge.score(Xtrain, Ytrain), ridge.score(Xtest, Ytest))

    # bc = load_breast_cancer()
    # bc = load_digits(n_class=2)
    bc = load_wine()
    X = bc['data']
    Y = bc['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)

    LogR = LogisticRegression().fit(Xtrain, Ytrain)
    print(LogR.score(Xtrain, Ytrain), LogR.score(Xtest, Ytest))

    Perp = Perceptron(gaussian_kernel = True).fit(Xtrain, Ytrain)
    print(Perp.score(Xtrain, Ytrain), Perp.score(Xtest, Ytest))


    print('done')