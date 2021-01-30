import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator
from sklearn.datasets import load_wine, load_breast_cancer, load_boston
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# First let's get the tree class right then check this
# Also note that the whole layout is a dictionary
# So not that seems each node need to have the following

# label: the features it is using at the moment
# threshold: the valve that this node is splitting
# gini: Current gini or entropy of the dataset at this node
# samples: number of samples at this node
# value: The number of each label at this node
# class: The final predict output of this node if it is a leaf


class DecisionTreeClassifier(BaseEstimator):
    def __init__(self,
                 criterion = 'gini',
                 splitter = 'best',
                 max_depth = 5,
                 min_samples_split = 1,
                 min_impurity_split = 0,
                 random_state = None
                ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split= min_samples_split
        self.Tree = None
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state

    def cal_proba(self, Y):
        Y_label = pd.Series(Y)
        return Y_label.value_counts()/Y_label.shape[0]

    def entropy(self, p):
        return - (p * np.log2(p)).sum()

    def gini(self, p):
        return 1 - np.power(p, 2).sum()

    def cal_impurity(self, Y):
        p = self.cal_proba(Y)
        return self.impurity_dict[self.criterion](p)

    def func_cal(self, sub_X, sub_Y, row):
        col, split = row.values
        return sub_Y.groupby(sub_X[:, col] > split).apply(lambda i: len(i) / len(sub_Y) * self.cal_impurity(i)).sum()

    def bestSplit(self, sub_X, sub_Y, split_df):
        Impurity_before_split = self.cal_impurity(sub_Y)
        if self.splitter == 'best':
            All_gain = Impurity_before_split - split_df.apply(lambda row : self.func_cal(sub_X, sub_Y, row), axis = 1)
            max_gain_index = All_gain.idxmax()
            split_decision = split_df.loc[max_gain_index, :]
            split_decision['best_gain'] = All_gain.max()
            left_df_index = sub_X[:, split_decision['Col_index']] <= split_decision['split']
            sub_X_left, sub_Y_left = sub_X[left_df_index], sub_Y[left_df_index]
            sub_X_right,sub_Y_right= sub_X[~left_df_index],sub_Y[~left_df_index]
            split_df = split_df[~((split_df['Col_index'] == split_decision['Col_index']) & (split_df['split'] == split_decision['split']))]
            return split_decision, split_df, sub_X_left, sub_X_right, sub_Y_left, sub_Y_right

        elif self.splitter == 'random':
            random_split_df = split_df.groupby('Col_index').apply(lambda df : df.sample(1)).reset_index(drop = True)
            All_gain = Impurity_before_split - random_split_df.apply(lambda row: self.func_cal(sub_X, sub_Y, row), axis=1)
            max_gain_index = All_gain.idxmax()
            split_decision = random_split_df.loc[max_gain_index, :]
            split_decision['best_gain'] = All_gain.max()
            left_df_index = sub_X[:, split_decision['Col_index']] <= split_decision['split']
            sub_X_left, sub_Y_left = sub_X[left_df_index], sub_Y[left_df_index]
            sub_X_right, sub_Y_right = sub_X[~left_df_index], sub_Y[~left_df_index]
            split_df = split_df[~((split_df['Col_index'] == split_decision['Col_index']) & (
                        split_df['split'] == split_decision['split']))]
            return split_decision, split_df, sub_X_left, sub_X_right, sub_Y_left, sub_Y_right

    def cal_mid(self, array):
        return (array[:-1] + array[1:]) / 2

    def init_params(self, X, Y):
        if type(X) is pd.DataFrame:
            self.X = X.values
        else:
            self.X = X

        self.total_number = X.shape[0]
        self.all_columns_gain = {i:0 for i in range(X.shape[1])}
        self.Y = pd.Series(np.ravel(np.array(Y)))
        self.impurity_dict = {'gini' : self.gini, 'entropy' : self.entropy}
        self.columns_unique = pd.DataFrame(self.X).apply(lambda col : [np.sort(col.unique())]).to_dict()
        self.columns_unique_dict = {i:self.cal_mid(self.columns_unique[i][0]) for i in self.columns_unique}
        self.split_df = pd.DataFrame(
            np.vstack([np.array([[i, j] for j in self.columns_unique_dict[i]], dtype = object) for i in self.columns_unique_dict]),
            columns = ['Col_index', 'split'])



    def buildTree(self, X, Y, split_df, node):
        node_raw_results = Y.value_counts().sort_index()
        node_results = node_raw_results / Y.shape[0]
        node.samples_counts = node_raw_results.values
        node.probas = node_results
        node.label = node_results.idxmax()

        if node.depth >= self.max_depth:
            node.leaf = True
            return node

        if X.shape[0] < self.min_samples_split:
            node.leaf = True
            return node

        if np.unique(Y).shape[0] == 1:
            node.leaf = True
            return node

        if X.shape[1] == 1:
            node.leaf = True
            return node

        split_decision, split_df, sub_X_left, sub_X_right, sub_Y_left, sub_Y_right = self.bestSplit(X, Y, split_df)

        if split_decision['best_gain'] < self.min_impurity_split:
            node.leaf = True
            return node

        node.threshold = split_decision['split']
        node.column = split_decision['Col_index']
        self.all_columns_gain[node.column] += Y.shape[0]/self.total_number * split_decision['best_gain']

        depth = node.depth
        node_left = Node_Classifier(depth = depth + 1)
        node_right= Node_Classifier(depth = depth + 1)

        node.left = self.buildTree(sub_X_left, sub_Y_left, split_df, node_left)
        node.right= self.buildTree(sub_X_right,sub_Y_right,split_df, node_right)
        return node

    def __predict_proba(self, X_, node):
        if node.leaf:
            return node.probas

        if X_[node.column] <= node.threshold:
            return self.__predict_proba(X_, node.left)
        return self.__predict_proba(X_, node.right)

    def _predict_proba(self, X_):
        return self.__predict_proba(X_, self.tree)

    def fit(self, X, Y):
        self.init_params(X, Y)
        node = Node_Classifier()
        self.tree = self.buildTree(self.X, self.Y, self.split_df, node)
        self.feature_importance = pd.Series(self.all_columns_gain)
        self.feature_importance = self.feature_importance / self.feature_importance.sum()
        self.feature_importance = self.feature_importance.values
        return self

    def predict_proba(self, X):
        Y_pred_proba = [self._predict_proba(X_) for X_ in X]
        Y_pred_proba = pd.concat(Y_pred_proba, axis = 1).fillna(0).T.values
        return Y_pred_proba

    def predict(self, X):
        return self.predict_proba(X).argmax(axis = 1)

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return accuracy_score(Y, Y_pred)

class Node_Classifier:
    def __init__(self, depth = 0):
        self.left  = None
        self.right = None

        self.column = None
        self.label = None
        self.threshold = None

        self.probas= None
        self.samples_counts = None
        self.depth = depth
        self.leaf  = False

    def display(self, wine = True):
        if wine:
            feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁',
                        '总酚', '类黄酮', '非黄烷类酚类', '花青素',
                        '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']

        print('The current node has the following properties:')
        if self.leaf:
            print('It is a leaf node')
        else:
            print('It is an internal node')
        print('The node has depth {}'.format(self.depth))
        print('The node is using Column index {} to split data'.format(self.column))
        if wine & (not self.leaf):
            print('Colname is {}'.format(feature_name[self.column]))
        print('The split threshold is {}'.format(self.threshold))
        print('The node has predict proba for each label {}'.format(self.probas))
        print('The node has samples {}'.format(self.samples_counts))
        print('The predict label output of this node is {}'.format(self.label))


class DecisionTreeRegressor(BaseEstimator):
    def __init__(self,
                 criterion = 'mse',
                 splitter = 'best',
                 max_depth = 5,
                 min_samples_split = 1,
                 min_impurity_split = 0,
                 random_state = None
                ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split= min_samples_split
        self.Tree = None
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state

    def cal_impurity(self, Y):
        Y_pred = np.ones(Y.shape) * Y.mean()
        return self.impurity_dict[self.criterion](Y, Y_pred)

    def func_cal(self, sub_X, sub_Y, row):
        col, split = row.values
        return sub_Y.groupby(sub_X[:, col] > split).apply(lambda i: len(i) / len(sub_Y) * self.cal_impurity(i)).sum()

    def bestSplit(self, sub_X, sub_Y, split_df):
        Impurity_before_split = self.cal_impurity(sub_Y)
        if self.splitter == 'best':
            All_gain = Impurity_before_split - split_df.apply(lambda row : self.func_cal(sub_X, sub_Y, row), axis = 1)
            max_gain_index = All_gain.idxmax()
            split_decision = split_df.loc[max_gain_index, :]
            split_decision['best_gain'] = All_gain.max()
            left_df_index = sub_X[:, split_decision['Col_index']] <= split_decision['split']
            sub_X_left, sub_Y_left = sub_X[left_df_index], sub_Y[left_df_index]
            sub_X_right,sub_Y_right= sub_X[~left_df_index],sub_Y[~left_df_index]
            split_df = split_df[~((split_df['Col_index'] == split_decision['Col_index']) & (split_df['split'] == split_decision['split']))]
            return split_decision, split_df, sub_X_left, sub_X_right, sub_Y_left, sub_Y_right

        elif self.splitter == 'random':
            random_split_df = split_df.groupby('Col_index').apply(lambda df : df.sample(1)).reset_index(drop = True)
            All_gain = Impurity_before_split - random_split_df.apply(lambda row: self.func_cal(sub_X, sub_Y, row), axis=1)
            max_gain_index = All_gain.idxmax()
            split_decision = random_split_df.loc[max_gain_index, :]
            split_decision['best_gain'] = All_gain.max()
            left_df_index = sub_X[:, split_decision['Col_index']] <= split_decision['split']
            sub_X_left, sub_Y_left = sub_X[left_df_index], sub_Y[left_df_index]
            sub_X_right, sub_Y_right = sub_X[~left_df_index], sub_Y[~left_df_index]
            split_df = split_df[~((split_df['Col_index'] == split_decision['Col_index']) & (
                        split_df['split'] == split_decision['split']))]
            return split_decision, split_df, sub_X_left, sub_X_right, sub_Y_left, sub_Y_right

    def cal_mid(self, array):
        return (array[:-1] + array[1:]) / 2

    def init_params(self, X, Y):
        if type(X) is pd.DataFrame:
            self.X = X.values
        else:
            self.X = X

        self.total_number = X.shape[0]
        self.all_columns_gain = {i:0 for i in range(X.shape[1])}
        self.Y = pd.Series(np.ravel(np.array(Y)))
        self.impurity_dict = {'mse' : mean_squared_error, 'mae' : mean_absolute_error}
        self.columns_unique = pd.DataFrame(self.X).apply(lambda col : [np.sort(col.unique())]).to_dict()
        self.columns_unique_dict = {i:self.cal_mid(self.columns_unique[i][0]) for i in self.columns_unique}
        self.split_df = pd.DataFrame(
            np.vstack([np.array([[i, j] for j in self.columns_unique_dict[i]], dtype = object) for i in self.columns_unique_dict]),
            columns = ['Col_index', 'split'])

    def buildTree(self, X, Y, split_df, node):
        # node_raw_results = Y.value_counts().sort_index()
        # node_results = node_raw_results / Y.shape[0]
        # node.samples_counts = node_raw_results.values
        # node.probas = node_results
        node.label = Y.mean()

        if node.depth >= self.max_depth:
            node.leaf = True
            return node

        if X.shape[0] < self.min_samples_split:
            node.leaf = True
            return node

        if np.unique(Y).shape[0] == 1:
            node.leaf = True
            return node

        if X.shape[1] == 1:
            node.leaf = True
            return node

        split_decision, split_df, sub_X_left, sub_X_right, sub_Y_left, sub_Y_right = self.bestSplit(X, Y, split_df)

        if split_decision['best_gain'] < self.min_impurity_split:
            node.leaf = True
            return node

        node.threshold = split_decision['split']
        node.column = split_decision['Col_index']
        self.all_columns_gain[node.column] += Y.shape[0]/self.total_number * split_decision['best_gain']

        depth = node.depth
        node_left = Node_Regressor(depth = depth + 1)
        node_right= Node_Regressor(depth = depth + 1)

        node.left = self.buildTree(sub_X_left, sub_Y_left, split_df, node_left)
        node.right= self.buildTree(sub_X_right,sub_Y_right,split_df, node_right)
        return node

    def fit(self, X, Y):
        self.init_params(X, Y)
        node = Node_Regressor()
        self.tree = self.buildTree(self.X, self.Y, self.split_df, node)
        self.feature_importance = pd.Series(self.all_columns_gain)
        self.feature_importance = self.feature_importance / self.feature_importance.sum()
        self.feature_importance = self.feature_importance.values
        return self

    def __predict(self, X_, node):
        if node.leaf:
            return node.label
        if X_[node.column] <= node.threshold:
            return self.__predict(X_, node.left)
        return self.__predict(X_, node.right)

    def _predict(self, X_):
        return self.__predict(X_, self.tree)

    def predict(self, X):
        Y_pred = np.array([self._predict(X_) for X_ in X])
        Y_pred[np.isnan(Y_pred)] = self.Y.mean()
        return Y_pred

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred)

class Node_Regressor:
    def __init__(self, depth = 0):
        self.left  = None
        self.right = None

        self.column = None
        self.label = None
        self.threshold = None

        self.depth = depth
        self.leaf  = False

    def display(self):
        print('The current node has the following properties:')
        if self.leaf:
            print('It is a leaf node')
        else:
            print('It is an internal node')
        print('The node has depth {}'.format(self.depth))
        print('The node is using Column index {} to split data'.format(self.column))
        print('The split threshold is {}'.format(self.threshold))
        print('The predict label output of this node is {}'.format(self.label))



if __name__ == '__main__':
    # data = pd.DataFrame(
    #     {'年龄': ['青年'] * 5 + ['中年'] * 5 + ['老年'] * 5,
    #      '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    #      '有自己的房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    #      '信贷情况': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    #      '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    #      })
    # oe = OrdinalEncoder()
    # X = data.iloc[:, :-1]
    # Y = data.iloc[:,  -1].replace({'是' : 1, '否' : 0})
    # X = oe.fit_transform(X)
    # model = DecisionTreeClassifier(max_depth = 3)
    # model = model.fit(X, Y)
    # print(model.score(X, Y))


    # data = load_breast_cancer()
    # data = load_wine()
    data = load_boston()
    X = data['data']
    Y = data['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, random_state = 42)
    # model = DecisionTreeClassifier(max_depth = 3
    #                                , splitter = 'random'
    #                                )
    model = DecisionTreeRegressor(max_depth = 20
                                  , splitter = 'best'
                                  )
    model = model.fit(Xtrain, Ytrain)
    print(model.score(Xtrain, Ytrain), model.score(Xtest, Ytest))
    pickle.dump(model, open('boston_model.pkl', 'wb'))

    print('done')