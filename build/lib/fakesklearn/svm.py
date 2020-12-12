import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import copy
from .common_tools import OneVsOneClassifier

# When setting paramters, name must be the same
# e.g. max_depth is a parameters in __init__
# THEN
# self.max_depth = max_depth
# could not do different names!!!!
# e.g. the following is wrong
# self.max_D = max_depth

class _SVC(BaseEstimator):
    def __init__(self, gamma=1e-2,
                 tol=1e-3, epsilon=1e-3,
                 gaussian_kernel=True,
                 max_iter=500, C=1,
                 predict_probability=True,
                 normalize=True
                 ):
        super().__init__()
        self.gamma = gamma
        self.tol = tol
        self.epsilon = epsilon
        self.gaussian_kernel = gaussian_kernel
        self.max_iter = max_iter
        self.C = C
        self.predict_probability = predict_probability
        self.normalize = normalize

    def init_params(self, X, Y):
        self.N, self.M = X.shape
        self.X = X.copy()
        self.Y = Y.copy()
        self.Y[self.Y == 0] = -1

        if self.normalize:
            self.stand = StandardScaler().fit(self.X)
            self.X = self.stand.transform(self.X)

        self.a = np.zeros(self.N)
        self.b = 0
        self.Gram_matrix = self.Kernel(self.X, self.X)
        self.E_list = self.cal_E_list()

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
        #                     if (i != j) and (j in range(N)) and (i in range(M)):
        #                         Gram_matrix[j, i] = Gram_matrix[i, j]
        else:
            Gram_matrix = X1 @ X2.T
        return Gram_matrix

    def _cal_single_E(self, i):
        Y_i = self.Y[i]
        K_ji = self.Gram_matrix[i, :]
        return (self.a * self.Y * K_ji).sum() + self.b - Y_i

    def cal_E_list(self):
        return pd.Series([self._cal_single_E(i) for i in range(self.N)])

    def bound_calculate(self, i, j):
        Y_i, Y_j = self.Y[i], self.Y[j]
        a_i, a_j = self.a[i], self.a[j]

        if Y_i != Y_j:
            L = max(0, a_j - a_i)
            H = min(self.C, self.C + a_j - a_i)
        else:
            L = max(0, a_i + a_j - self.C)
            H = min(self.C, a_i + a_j)
        return L, H

    def check_violate(self, i):
        E_i = self.E_list[i]
        Y_i = self.Y[i]
        result_i = E_i * Y_i
        a_i = self.a[i]
        return ((result_i < - self.tol) & (a_i < self.C)) | ((result_i > self.tol) & (a_i > self.C))

    def selecting_second(self, i):
        E_i = self.E_list[i]
        index_list_without_i = list(range(self.N))
        index_list_without_i.pop(i)
        E_diff = np.abs(self.E_list[index_list_without_i] - E_i)
        #         if E_diff.sum() != 0:
        #             E_max_diff = (E_diff == E_diff.max())
        #             E_max_diff = E_max_diff[E_max_diff]
        #             j = np.random.choice(E_max_diff.index)
        #         else:
        #             j = np.random.choice(E_diff.index)
        j = E_diff.idxmax()
        return j

    def update(self, i, j):
        L, H = self.bound_calculate(i, j)

        E_i, E_j = self.E_list[i], self.E_list[j]
        Y_i, Y_j = self.Y[i], self.Y[j]
        a_i, a_j = self.a[i], self.a[j]

        if L == H:
            return False

        kii = self.Gram_matrix[i, j]
        kjj = self.Gram_matrix[j, j]
        kij = self.Gram_matrix[i, j]
        det = kii + kjj - 2 * kij

        if det < 0:
            return False

        a_j_new_unc = a_j + Y_j * (E_i - E_j) / det

        if a_j_new_unc <= L:
            a_j_new = L
        elif a_j_new_unc >= H:
            a_j_new = H
        else:
            a_j_new = a_j_new_unc

        if abs(a_j - a_j_new) < self.epsilon * (a_j + a_j_new + self.epsilon):
            #        if abs(a_j - a_j_new) < self.epsilon * (a_j + a_j_new):
            #        if abs(a_j - a_j_new) < self.epsilon:
            return False

        #         print('i, j, b:', i, j, self.b)
        #         print('a_i_old, a_j_old: ', a_i, a_j)
        #         print('L, H:', L, H)
        #         print('a_j_new_unc:', a_j_new_unc)
        #         print('{} * ({} - {} * {})'.format(Y_i, a_i * Y_i + a_j * Y_j, a_j_new, Y_j))
        a_i_new = Y_i * (a_i * Y_i + a_j * Y_j - a_j_new * Y_j)

        #         print('a_i_new, a_j_new: ', a_i_new, a_j_new)

        self.b = -self.E_list[i] - Y_i * kii * (a_i_new - a_i) - \
                 Y_j * kij * (a_j_new - a_j) + self.b

        self.a[i] = a_i_new
        self.a[j] = a_j_new
        self.E_list = self.cal_E_list()
        return True

    def iterate(self, indexes):
        for i in indexes:
            if self.check_violate(i):
                j = self.selecting_second(i)
                update_success = self.update(i, j)
                if update_success:
                    # print('points {}, {} updated successfully, are now {}, {}'.format(i, j, self.a[i], self.a[j]))
                    return False
        return True

    def fit(self, X, Y):
        self.init_params(X, Y)
        finish = False
        self.step = 0
        while (self.step <= self.max_iter) and (not finish):
            self.step += 1
            margin_points_idx = np.where((self.a > 0) & (self.a < 0))[0]
            if len(margin_points_idx) > 0:
                finish = self.iterate(margin_points_idx)
            else:
                finish = self.iterate(range(self.N))

        if self.predict_probability:
            X_log = self.decision_function(self.X).reshape(-1, 1)
            self.std = StandardScaler().fit(X_log)
            X_log = self.std.transform(X_log)
            self.logistic_model = LogisticRegression(solver='lbfgs').fit(X_log, self.Y)

        return self

    def _predict_dis(self, col):
        return np.sum(self.a * self.Y * col) + self.b

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
            return Y_pred_proba.argmax(axis=1)
        Y_pred = np.sign(self.decision_function(X))
        Y_pred[Y_pred == -1] = 0
        return Y_pred

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(Y, Y_pred)

class SVC(BaseEstimator, OneVsOneClassifier):
    def __init__(self, gamma=1e-2,
                 tol=1e-3, epsilon=1e-3,
                 gaussian_kernel=True,
                 max_iter=500, C=1,
                 predict_probability=True,
                 normalize=True
                 ):
        super().__init__()
        self.gamma = gamma
        self.tol = tol
        self.epsilon = epsilon
        self.gaussian_kernel = gaussian_kernel
        self.max_iter = max_iter
        self.C = C
        self.predict_probability = predict_probability
        self.normalize = normalize
        self.binary = True

        estimator_params = ('gamma', 'tol', 'epsilon', 'gaussian_kernel', 'max_iter', 'C',
                            'predict_probability', 'normalize')
        self.base_estimator_ = _SVC(**{p: getattr(self, p) for p in estimator_params})

if __name__ == '__main__':
    # data = load_breast_cancer()
    data = load_wine()
    # data = load_digits(n_class = 3)
    # data = load_iris()
    X = data['data']
    Y = data['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3,random_state=42)

    # random_state = 10, last one is very high
    # model = SVC(gaussian_kernel=False, predict_probability=False, normalize=False)
    # model.fit(Xtrain, Ytrain)
    # print(model.score(Xtrain, Ytrain), model.score(Xtest, Ytest))
    #
    # model_std = SVC(gaussian_kernel=False, predict_probability=False, normalize=True)
    # model_std.fit(Xtrain, Ytrain)
    # print(model_std.score(Xtrain, Ytrain), model_std.score(Xtest, Ytest))
    # model = SVC(gaussian_kernel=False, predict_probability=True, normalize=False)
    # model.fit(Xtrain, Ytrain)
    # print(model.score(Xtrain, Ytrain), model.score(Xtest, Ytest))
    # model_std = SVC(gaussian_kernel=False, predict_probability=True, normalize=True)
    # model_std.fit(Xtrain, Ytrain)
    # print(model_std.score(Xtrain, Ytrain), model_std.score(Xtest, Ytest))
    # model = SVC(gaussian_kernel=True, predict_probability=True, normalize=False)
    # model.fit(Xtrain, Ytrain)
    # print(model.score(Xtrain, Ytrain), model.score(Xtest, Ytest))
    #
    model_std = SVC(gaussian_kernel=True, predict_probability=True, normalize=True)
    model_std.fit(Xtrain, Ytrain)
    print(model_std.score(Xtrain, Ytrain), model_std.score(Xtest, Ytest))



    print('done')
