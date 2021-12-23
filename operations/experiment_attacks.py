import numpy as np
import pandas as pd
from data import *
from models import *
from samplers import *
from attacks import *
from inference import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':

    n_exp = 1
    n_samples = 70
    n_cov = 3

    X, y = get_spam_data("data/uciData.csv")


    for i in range(n_exp):
        clf = LogisticRegression(penalty='l1', C=0.1, solver='saga')
        clf.fit(X_train, y_train)
        weights = np.abs(clf.coef_)
        S = []
        for w in weights:
            S.append( (-w).argsort()[:n_cov] )
        S = np.concatenate( S, axis=0 )
        S = np.unique( S )

        params = {
                    "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                    "k"      : 2,     # Number of classes
                    "var"    : 0.1,   # Proportion of max variance of betas
                    "ut"     :  np.array([[1.0, 0.0],[0.0, 1.0]]), # Ut matrix for defender
                    "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                                # what the classifier says, columns
                                                # real label!!!
                    "sampler_star" : lambda x: sample_original_instance_star(x,
                     n_samples=40, rho=1, x=None, mode='sample', heuristic='uniform'),
                     ##
                     "clf" : clf,
                     "tolerance" : 0, # For ABC
                     "classes" : np.array([0,1]),
                     "S"       : S, # Set of index representing covariates with
                                                 # "sufficient" information
                     "X_train"   : X_train,
                     "distance_to_original" : 1 # Numbers of changes allowed to adversary
                }

        print('Adversary Unaware Accuracy Clean Data', accuracy_score(y_test, clf.predict(X_test)))

        ## Attack test set
        attack_ARA(X[0], y[0], params)
        X_att = attack_set(X_test, y_test, params)

        print('Adversary Unaware Accuracy Attacked Data', accuracy_score(y_test, clf.predict(X_att)))



    df = pd.DataFrame({"tolerance":tolerance_grid, "acc_raw_clean":acc_raw_clean,
     "acc_raw_att":acc_raw_att, "acc_acra_att":acc_acra_att})
    print('Writing Experiment ', i)
    name = "results/exp_tolerance/" + "exp_tolerance" + str(i) + ".csv"
    df.to_csv(name, index=False)
