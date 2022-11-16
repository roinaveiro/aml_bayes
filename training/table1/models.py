import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.inspection import permutation_importance
from sklearn import svm
from data import *

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import eli5
    from eli5.sklearn import PermutationImportance

'''
Train discriminative classifiers and obtain most important covariates
'''

def get_top_featues(X, y, clf):

    perm = PermutationImportance(clf, random_state=1).fit(X, y)
    return np.argsort(-perm.feature_importances_)

def SGLD_noise(coef, lr):
    return coef + np.sqrt(2*lr)*np.random.randn(*coef.shape)


def train_clf(X_train, y_train, n_cov, flag='lr'):

    if flag == 'sgd_lr':
        lr = 0.005
        clf = SGDClassifier(penalty='l1', learning_rate='constant', eta0 = lr, loss='log')
        clf.fit(X_train, y_train)
        weights = np.abs(clf.coef_)
        S = []
        for w in weights:
            S.append( (-w).argsort()[:n_cov] )
        S = np.concatenate( S, axis=0 )
        S = np.unique( S )
        return clf, S

    if flag == 'bayes_lr':
        lr = 0.005
        n_est = 5
        base_clf = SGDClassifier(penalty='l1', learning_rate='constant', eta0 = lr, loss='log')
        clf = BaggingClassifier(base_clf, n_estimators=n_est)
        clf.fit(X_train, y_train)

        for i in range(n_est):
            clf.estimators_[i].coef_ = SGLD_noise(clf.estimators_[i].coef_, lr)

        weights = np.abs(np.mean(np.asarray([clf.estimators_[i].coef_ for i in range(n_est)] ), axis=0 ))

        S = []
        for w in weights:
            S.append( (-w).argsort()[:n_cov] )
        S = np.concatenate( S, axis=0 )
        S = np.unique( S )
        return clf, S

    if flag == 'lr':
        clf = LogisticRegression(penalty='l1', C=0.1, solver='saga')
        clf.fit(X_train, y_train)
        weights = np.abs(clf.coef_)
        S = []
        for w in weights:
            S.append( (-w).argsort()[:n_cov] )
        S = np.concatenate( S, axis=0 )
        S = np.unique( S )
        return clf, S

    # if flag == 'rf':
    #     clf = RandomForestClassifier()
    #     clf.fit(X_train,y_train)
    #     result = permutation_importance(clf, X_train, y_train, n_repeats=10)
    #     sorted_idx = np.argsort(-result.importances_mean)
    #     S = sorted_idx[:n_cov]
    #     return clf, S

    if flag == 'rf':
        clf = RandomForestClassifier()
        clf.fit(X_train,y_train)
        sorted_idx = get_top_featues(X_train, y_train, clf=clf)
        S = sorted_idx[:n_cov]
        return clf, S

    if flag == 'nn':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(3, 2), random_state=1)
        clf.fit(X_train,y_train)
        sorted_idx = get_top_featues(X_train, y_train, clf=clf)
        S = sorted_idx[:n_cov]
        return clf, S

    if flag == 'adam_nn':
        lr = 0.001
        clf = MLPClassifier(solver='adam', alpha=1e-5,learning_rate_init=lr,
                     hidden_layer_sizes=(3, 2), max_iter=500)
        clf.fit(X_train,y_train)
        sorted_idx = get_top_featues(X_train, y_train, clf=clf)
        S = sorted_idx[:n_cov]
        return clf, S

    if flag == 'bayes_nn':
        lr = 0.001
        n_est = 5
        base_clf = MLPClassifier(solver='adam', alpha=1e-5,learning_rate_init=lr,
                     hidden_layer_sizes=(3, 2), max_iter=500)
        clf = BaggingClassifier(base_clf, n_estimators=n_est)
        clf.fit(X_train,y_train)

        for i in range(n_est):
            for l in clf.estimators_[i].coefs_:
                l = SGLD_noise(l, lr)

        sorted_idx = get_top_featues(X_train, y_train, clf=clf)
        S = sorted_idx[:n_cov]
        return clf, S

    if flag == 'svm':
        clf = svm.SVC(probability=True)
        clf.fit(X_train, y_train)
        sorted_idx = get_top_featues(X_train, y_train, clf=clf)
        S = sorted_idx[:n_cov]
        return clf, S

    if flag == 'nb':
        clf = BernoulliNB(alpha=1.0e-10)
        clf.fit(X_train, y_train)
        sorted_idx = get_top_featues(X_train, y_train, clf=clf)
        S = sorted_idx[:n_cov]
        return clf, S


if __name__ == '__main__':
    X, y = get_spam_data("data/uciData.csv")
    X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.3)
    flag = 'svm'
    clf, S = train_clf(X_train, y_train, 11, flag=flag)
    #pr = clf.predict(X_test)
    print( clf.predict_proba(X_test) )
