#Author : Xiangyang Ni
#Email  : frank935460794@live.com

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from fakesklearn import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('For Regression Methods, the RandomForest, GBDT are used for the Boston DataSets')
    print('The following are the results')
    boston = load_boston()
    X = boston['data']
    Y = boston['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state = 42)
    GBR = GBDTRegressor(max_depth=3, n_estimators=100,
                        learning_rate=0.05, subsample=0.85
                        ).fit(Xtrain, Ytrain)
    GBR_train_score, GBR_test_score = np.around(GBR.score(Xtrain, Ytrain), 2), np.around(GBR.score(Xtest, Ytest), 2)
    RFR = RandomForestRegressor(max_depth = 5, n_estimators = 100, subsample = 0.85).fit(Xtrain, Ytrain)
    RFR_train_score, RFR_test_score = np.around(RFR.score(Xtrain, Ytrain), 2), np.around(RFR.score(Xtest, Ytest), 2)
    result_rgr = pd.DataFrame({'Train_R2' : [RFR_train_score, GBR_train_score],
                           'Test_R2' : [RFR_test_score, GBR_test_score]})
    result_rgr.index = ['RandomForest', 'GBDT']
    print('\n')
    print('=======================================================================================')
    print(result_rgr)
    print('=======================================================================================')
    print('\n')
    print('For Classification methods, the Adaboost, RandomForest, GBDT, SVM are used for the breast cancer Datasets')
    bc = load_breast_cancer()
    X = bc['data']
    Y = bc['target']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state = 42)

    RFC = RandomForestClassifer(max_depth = 5, n_estimators = 100, subsample = 0.85).fit(Xtrain, Ytrain)
    RFC_train_score, RFC_test_score = np.around(RFC.score(Xtrain, Ytrain), 2), np.around(RFC.score(Xtest, Ytest))

    ADC = AdaBoostClassifer(max_depth = 3, n_estimators = 100).fit(Xtrain, Ytrain)
    ADC_train_score, ADC_test_score = np.around(ADC.score(Xtrain, Ytrain), 2), np.around(ADC.score(Xtest, Ytest))

    GBC = GBDTClassifier(max_depth = 7, n_estimators = 100, learning_rate = 0.1, subsample = 0.85).fit(Xtrain, Ytrain)
    GBC_train_score, GBC_test_score = np.around(GBC.score(Xtrain, Ytrain), 2), np.around(GBC.score(Xtest, Ytest))

    svc = SVC(gamma = 1e2).fit(Xtrain, Ytrain)
    svc_train_score, svc_test_score = np.around(svc.score(Xtrain, Ytrain), 2), np.around(svc.score(Xtest, Ytest))

    result_clf = pd.DataFrame({'Train_acc': [RFC_train_score, ADC_train_score, GBC_train_score, svc_train_score],
                               'Test_acc' : [RFC_test_score,  ADC_test_score,  GBC_test_score,  svc_test_score ]
                               })

    result_clf.index = ['RandomForest', 'AdaBoost', 'GBDT', 'SVM']
    print('\n')
    print('=======================================================================================')
    print(result_clf)
    print('=======================================================================================')
    print('\n')
    print('done')

