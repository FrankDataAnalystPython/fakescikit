import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.tree._classes import BaseDecisionTree
from sklearn.base import clone
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
#warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_boston

class GBDTRegressor(BaseEstimator):
    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=200,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 random_state=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 presort='deprecated',
                 ccp_alpha=0.0,
                 tol=1e-4):

        super().__init__()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.presort = presort
        self.tol = tol

    def init_params(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
        if self.random_state is None:
            self.random_state = int(time.time())
        estimator_params = ("criterion", "max_depth", "min_samples_split",
                            "min_samples_leaf", "min_weight_fraction_leaf",
                            "max_features", "max_leaf_nodes",
                            "min_impurity_decrease", "min_impurity_split",
                            "random_state", "ccp_alpha")
        self.base_estimator_ = DecisionTreeRegressor(**{p: getattr(self, p)
                                                        for p in estimator_params
                                                        })
        self.rng = np.random.RandomState(self.random_state)
        seed_list = self.rng.choice(range(10000), size=self.n_estimators,
                                    replace=True
                                    )
        self.estimators_ = [clone(self.base_estimator_.set_params(random_state=i))
                            for i in seed_list]

    def fit(self, X, Y):
        self.init_params(X, Y)
        subsample = self.subsample
        Y_predict = 0
        rest_error = Y
        if subsample < 1:
            subsample_size = int(X.shape[0] * self.subsample)

        for i in range(len(self.estimators_)):
            if subsample < 1:
                random_index = self.rng.choice(range(X.shape[0]), size=subsample_size, replace=False)
                tmp_X, tmp_error = X[random_index, :], rest_error[random_index]
            else:
                tmp_X, tmp_error = X, rest_error
            base_model = self.estimators_[i].fit(tmp_X, tmp_error)
            Y_predict += self.learning_rate * base_model.predict(X)
            rest_error = Y - Y_predict
            # self.estimators_[i] = base_model
        self.feature_importances_ = np.array([i.feature_importances_ for i in self.estimators_]).mean(axis = 0)
        return self

    def predict(self, X):
        error_pred_list = np.array([model.predict(X) for model in self.estimators_])
        Y_predict = self.learning_rate * error_pred_list.sum(axis=0)
        return Y_predict

    def score(self, X, Y):
        Y_predict = self.predict(X)
        return r2_score(Y, Y_predict)


def p2logodds(p):
    odds = p / (1 - p)
    return np.log(odds)


def logodds2p(logodds):
    exp_logodds = np.exp(logodds)
    return exp_logodds / (1 + exp_logodds)


def logodds_error_pred_function(sub_df):
    return sub_df['error'].sum() / (sub_df['p_values'] * (1 - sub_df['p_values'])).sum()


class GBDTClassifier(BaseEstimator):
    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 random_state=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 presort='deprecated',
                 ccp_alpha=0.0,
                 ):

        super().__init__()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.presort = presort


    def init_params(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
        if self.random_state is None:
            self.random_state = int(time.time())

        estimator_params = ("criterion", "max_depth", "min_samples_split",
                            "min_samples_leaf", "min_weight_fraction_leaf",
                            "max_features", "max_leaf_nodes",
                            "min_impurity_decrease", "min_impurity_split",
                            "random_state", "ccp_alpha")
        self.base_estimator_ = DecisionTreeRegressor(**{p: getattr(self, p)
                                                        for p in estimator_params
                                                        })
        self.rng = np.random.RandomState(self.random_state)
        seed_list = self.rng.choice(range(10000), size=self.n_estimators,
                                    replace=True
                                    )
        self.estimators_ = [clone(self.base_estimator_.set_params(random_state=i))
                            for i in seed_list]
        self.leaf_logodds_pred_ = [[] for i in seed_list]


    def fit(self, X, Y):
        self.init_params(X, Y)
        subsample = self.subsample
        self.Y_predict_initial_logodds = p2logodds(Y.mean())
        Y_predict_P = np.array([Y.mean()] * X.shape[0])
        Y_predict_logodds = np.array([self.Y_predict_initial_logodds] * X.shape[0])
        # 计算概率残差
        rest_error = Y - Y_predict_P

        if subsample < 1:
            subsample_size = int(X.shape[0] * self.subsample)

        for i in range(len(self.estimators_)):
            if subsample < 1:
                random_index = self.rng.choice(range(X.shape[0]), size=subsample_size, replace=False)
                tmp_X, tmp_error, tmp_p = X[random_index, :], rest_error[random_index], Y_predict_P[random_index]
            else:
                tmp_X, tmp_error, tmp_p = X, rest_error, Y_predict_P

            # 训练
            base_model = self.estimators_[i].fit(tmp_X, tmp_error)

            # 将训练集里面所有的数据所在的叶节点的位置取出来，稍后需要做输出
            all_leaf_samples = base_model.apply(X)

            # 计算当前模型下每个叶节点输出，注意，如果subsample小于1，则只针对小subsample来算叶节点的输出值
            if subsample < 1:
                leaf_samples = all_leaf_samples[random_index]
            else:
                leaf_samples = all_leaf_samples

            # 记录上一次迭代中所有样本的概率和算出的error还有在哪个叶节点记录下来
            tmp_df = pd.DataFrame({'p_values': tmp_p,
                                   'error': tmp_error,
                                   'leaf': leaf_samples
                                   })
            # 使用groupby计算各个节点的输出，以series的形式保留下来
            logodds_error_pred = tmp_df.groupby('leaf').apply(logodds_error_pred_function)

            # 记录到leaf_logodds_pred_里去
            self.leaf_logodds_pred_[i] = logodds_error_pred

            # 可以接使用索引的方式将所有样本该输出的logodds_error索引出来乘上学习率加到之前的Y_predict_logodds里面去
            Y_predict_logodds += self.learning_rate * logodds_error_pred[all_leaf_samples].values

            # 每个样本转换出类概率的值为下一次迭代做准备
            Y_predict_P = logodds2p(Y_predict_logodds)

            # 计算每个样本在这一次迭代下剩余的残差，为下一次迭代做准备
            rest_error = Y - Y_predict_P
        self.feature_importances_ = np.array([i.feature_importances_ for i in self.estimators_]).mean(axis=0)
        return self

    def predict_prob(self, X):
        # 首先将初始值拿过来
        Y_predict_logodds = np.array([self.Y_predict_initial_logodds] * X.shape[0])

        # 所有的模型，对所有的样本来apply记录每个样本落在了哪个叶节点上
        Total_estimator_leaf_samples = [model.apply(X) for model in self.estimators_]

        # 一对一对的，每个样本在每个模型的叶节点的输出，放入，self.leaf_logodds_pred_里记录的每个叶节点应该输出的logodds
        all_error_logodds = np.array([logodds_error_pred[all_leaf_samples].values
                                      for logodds_error_pred, all_leaf_samples in
                                      zip(self.leaf_logodds_pred_, Total_estimator_leaf_samples)])

        # 叠加求和并且加上学习率和初始值
        Y_predict_logodds += 0.1 * all_error_logodds.sum(axis=0)

        # 计算类概率的值
        Y_predict_prob = logodds2p(Y_predict_logodds)

        # 输出和二分类classifier差不多的pred_prob的结果
        return np.hstack(((1 - Y_predict_prob).reshape(-1, 1), Y_predict_prob.reshape(-1, 1)))

    def predict(self, X):
        Y = self.predict_prob(X)[:, -1]
        return (Y >= 0.5).astype(int)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(Y, Y_pred)

class RandomForestRegressor(BaseEstimator):
    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 random_state=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 presort='deprecated',
                 ccp_alpha=0.0,
                 ):

        super().__init__()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.presort = presort


    def init_params(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
        if self.random_state is None:
            self.random_state = int(time.time())

        estimator_params = ("criterion", "max_depth", "min_samples_split",
                            "min_samples_leaf", "min_weight_fraction_leaf",
                            "max_features", "max_leaf_nodes",
                            "min_impurity_decrease", "min_impurity_split",
                            "random_state", "ccp_alpha")
        self.base_estimator_ = DecisionTreeRegressor(**{p: getattr(self, p)
                                                        for p in estimator_params
                                                        })
        self.rng = np.random.RandomState(self.random_state)
        seed_list = self.rng.choice(range(10000), size=self.n_estimators,
                                    replace=True
                                    )
        self.estimators_ = [clone(self.base_estimator_.set_params(random_state=i))
                            for i in seed_list]


    def fit(self, X, Y):
        self.init_params(X, Y)
        subsample = self.subsample
        if subsample < 1:
            subsample_size = int(X.shape[0] * self.subsample)

        for i in range(len(self.estimators_)):
            if subsample < 1:
                random_index = self.rng.choice(range(X.shape[0]), size=subsample_size, replace=False)
                tmp_X, tmp_Y = self.X[random_index, :], self.Y[random_index]
            else:
                tmp_X, tmp_Y = self.X, self.Y
            # 训练
            self.estimators_[i].fit(tmp_X, tmp_Y)

        self.feature_importances_ = np.array([i.feature_importances_ for i in self.estimators_]).mean(axis = 0)
        return self

    def predict(self, X):
        return  np.array([i.predict(X) for i in self.estimators_]).mean(axis = 0)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred)

class RandomForestClassifer(BaseEstimator):
    def __init__(self,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='gini',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=5,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 random_state=None,
                 max_features=None,
                 max_leaf_nodes=None,
                 presort='deprecated',
                 ):

        super().__init__()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.presort = presort


    def init_params(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
        if self.random_state is None:
            self.random_state = int(time.time())

        estimator_params = ("criterion", "max_depth", "min_samples_split",
                            "min_samples_leaf", "min_weight_fraction_leaf",
                            "max_features", "max_leaf_nodes",
                            "min_impurity_decrease", "min_impurity_split",
                            "random_state")
        self.base_estimator_ = DecisionTreeClassifier(**{p: getattr(self, p)
                                                        for p in estimator_params
                                                        })
        self.rng = np.random.RandomState(self.random_state)
        seed_list = self.rng.choice(range(10000), size=self.n_estimators,
                                    replace=True
                                    )
        self.estimators_ = [clone(self.base_estimator_.set_params(random_state=i))
                            for i in seed_list]


    def fit(self, X, Y):
        self.init_params(X, Y)
        subsample = self.subsample
        if subsample < 1:
            subsample_size = int(X.shape[0] * self.subsample)

        for i in range(len(self.estimators_)):
            if subsample < 1:
                random_index = self.rng.choice(range(self.X.shape[0]), size=subsample_size, replace=True)
                tmp_X, tmp_Y = self.X[random_index, :], self.Y[random_index]
            else:
                tmp_X, tmp_Y = self.X, self.Y
            # 训练
            self.estimators_[i].fit(tmp_X, tmp_Y)

        self.feature_importances_ = np.array([i.feature_importances_ for i in self.estimators_]).mean(axis = 0)
        return self

    def predict_proba(self, X):
        return np.array([i.predict_proba(X) for i in self.estimators_]).mean(axis = 0)


    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis = 1)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(Y, Y_pred)


if __name__ == '__main__':
    boston = load_boston()
    X = boston['data']
    Y = boston['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)
    gbdtregressor = GBDTRegressor(max_depth=3, n_estimators=100,
                                  learning_rate=0.05, subsample=0.85,
                                  random_state=42)
    gbdtregressor = gbdtregressor.fit(Xtrain, Ytrain)
    print(gbdtregressor.score(Xtrain, Ytrain), gbdtregressor.score(Xtest, Ytest))

    rfr = RandomForestRegressor(max_depth=10, n_estimators=100,
                                learning_rate=0.05, subsample=0.9)
    rfr = rfr.fit(Xtrain, Ytrain)
    print(rfr.score(Xtrain, Ytrain), rfr.score(Xtest, Ytest))

    bc = load_breast_cancer()
    X = bc['data']
    Y = bc['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state = 42)

    rfc = RandomForestClassifer(max_depth=5, n_estimators=100, subsample=0.9)
    rfc = rfc.fit(Xtrain, Ytrain)
    print(rfc.score(Xtrain, Ytrain), rfc.score(Xtest, Ytest))

    gbdtclf = GBDTClassifier(max_depth=3, n_estimators=100,
                             learning_rate=0.05, subsample=0.85,
                             random_state=42)
    gbdtclf = gbdtclf.fit(Xtrain, Ytrain)
    print(gbdtclf.score(Xtrain, Ytrain), gbdtclf.score(Xtest, Ytest))

    print('done')




