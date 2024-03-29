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


if __name__ == '__main__':

    n_exp = 10
    n_samples = 40
    tolerance_grid = [1,2,3,4,5,6,7,8]
    n_cov = 11
    var_l = 0.1
    flag = "rf"


    # flag_grid = ['nb', 'lr', 'rf', 'nn']
    save = True

    #X, y = get_sentiment_data("data/clean_imdb_sent.csv")
    X, y = get_sentiment_data("data/clean_imdb_sent_2.csv")


    for i in range(n_exp):

        print('Experiment: ', i)

        acc_raw_clean = np.zeros(len(tolerance_grid))
        acc_raw_att = np.zeros(len(tolerance_grid))
        acc_acra_att = np.zeros(len(tolerance_grid))

        X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.1)
        clf, S = train_clf(X_train, y_train, n_cov, flag)
        
        for j, tolerance in enumerate(tolerance_grid):

            acc_raw_clean[j] = accuracy_score(y_test, clf.predict(X_test))
            print( str(flag) + "accuracy on clean data", acc_raw_clean[j] )

        
            params = {
                        "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                        "k"      : 2,     # Number of classes
                        "var"    : var_l,   # Proportion of max variance of betas
                        "ut"     :  np.array([[1.0, 0.0],[0.0, 1.0]]), # Ut matrix for defender
                        "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                                    # what the classifier says, columns
                                                    # real label!!!
                        "sampler_star" : lambda x: sample_original_instance_star(x,
                        n_samples=100, rho=1, x=None, mode='sample', heuristic='uniform'),
                        ##
                        "clf" : clf,
                        "tolerance" : tolerance, # For ABC
                        "classes" : np.array([0,1]),
                        "S"       : S, # Set of index representing covariates with
                                                    # "sufficient" information
                        "X_train"   : X_train,
                        "distance_to_original" : 2, # Numbers of changes allowed to adversary
                        "stop" : True, # Stopping condition for ABC
                        "max_iter": 500, ## max iterations for ABC when stop=True
                        "dev": 0.25

                    }

            print(n_samples)
            
            X_att = attack_set(X_test, y_test, params)
            acc_raw_att[j] = accuracy_score(y_test, clf.predict(X_att))
            print( str(flag) + "accuracy on tainted data", acc_raw_att[j])

            sampler = lambda x: sample_original_instance(x, n_samples= n_samples, params = params)
            pr = parallel_predict_aware(X_att, sampler, params)
            acc_acra_att[j] =  accuracy_score(y_test, pr)
            print( str(flag) + "ACRA accuracy on tainted data var " + str(params["var"]), acc_acra_att[j])


    
        df = pd.DataFrame({"tolerance":tolerance_grid, "acc_raw_clean":acc_raw_clean,
        "acc_raw_att":acc_raw_att, "acc_acra_att":acc_acra_att})
        print(df)
        
        if save:
            print('Writing Experiment ', i)
            name = "results/sentiment_2/exp_tolerance/" + "exp_tolerance_" + str(i) + ".csv"
            df.to_csv(name, index=False)
