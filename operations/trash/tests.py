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

X, y = get_sentiment_data("data/clean_imdb_sent.csv")
X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.2)
n_cov = 11
n_samples = 20

clf = LogisticRegression(penalty='l1', C=1.0, solver='saga')
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
                        "var"    : 0.0,   # Proportion of max variance of betas
                        "ut"     :  np.array([[1.0, 0.0],[0.0, 1.0]]), # Ut matrix for defender
                        "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                                    # what the classifier says, columns
                                                    # real label!!!
                        "sampler_star" : lambda x: sample_original_instance_star(x,
                         n_samples=40, rho=1, x=None, mode='sample', heuristic='uniform'),
                         ##
                         "clf" : clf,
                         "tolerance" : 1, # For ABC
                         "classes" : np.array([0,1]),
                         "S"       : S, # Set of index representing covariates with
                                                     # "sufficient" information
                         "X_train"   : X_train,
                         "distance_to_original" : 2, # Numbers of changes allowed to adversary
                         "stop" : True, # Stopping condition for ABC
                         "max_iter": 1000, ## max iterations for ABC when stop=True
                         "dev": 0.5

                    }

print('Adversary Unaware Accuracy Clean Data', accuracy_score(y_test, clf.predict(X_test)))

## Attack test set
attack_ARA(X[0], y[0], params)
X_att = attack_set(X_test, y_test, params)

print('Adversary Unaware Accuracy Attacked Data', accuracy_score(y_test, clf.predict(X_att)))

sampler = lambda x: sample_original_instance(x, n_samples= n_samples, params = params)
pr = parallel_predict_aware(X_att, sampler, params)
acc_acra_att =  accuracy_score(y_test, pr)
print( "ACRA accuracy on tainted data", acc_acra_att)
