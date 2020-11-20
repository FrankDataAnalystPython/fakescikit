# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from fakesklearn import GBDTRegressor, GBDTClassifier, SVC

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    boston = load_boston()
    X = boston['data']
    Y = boston['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state = 42)
    gbdtregressor = GBDTRegressor(max_depth=3, n_estimators=100,
                         learning_rate=0.05, subsample=0.85,
                         random_state=42
                         )
    gbdtregressor = gbdtregressor.fit(Xtrain, Ytrain)
    print('boston, gbdtregressor')
    print(gbdtregressor.score(Xtrain, Ytrain), gbdtregressor.score(Xtest, Ytest))

    bc = load_breast_cancer()
    X = bc['data']
    Y = bc['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state = 41)
    gbdtclf = GBDTClassifier(n_estimators = 100, learning_rate = 0.2)
    gbdtclf = gbdtclf.fit(Xtrain, Ytrain)
    print('bc, gbdtclassifier')
    print(gbdtclf.score(Xtrain, Ytrain), gbdtclf.score(Xtest, Ytest))

    svc = SVC(gaussian_kernel=True, predict_probability=True, normalize=True,
              max_itera = 500, tol = 1e-3, gamma = 1e-2
              )
    svc = svc.fit(Xtrain, Ytrain)
    print('bc, svm')
    print(svc.score(Xtrain, Ytrain), svc.score(Xtest, Ytest))

    print('done')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
