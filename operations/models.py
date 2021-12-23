import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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


def train_clf(X_train, y_train, n_cov, flag='lr'):

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
    #
