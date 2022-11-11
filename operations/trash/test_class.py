import numpy as np
import pandas as pd
from data import *
from models import train_clf
from samplers import *
from attacks import *
from inference import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


if __name__ == '__main__':

    n_exp = 10
    n_samples = 40
    tolerance = 1
    n_cov = 11
    var_l = 0.1
    flag_grid = ['nn', 'gb', 'rf', 'adaboost', 'svm']

    q = 0.1 #Percentage for test

    # X, y = get_spam_data("data/uciData.csv")
    X, y = get_sentiment_data("data/clean_imdb_sent_2.csv")


    for i in range(n_exp):

        print('Experiment: ', i)


        X_train, X_test, y_train, y_test = generate_train_test(X, y, q=q)

        for j, flag in enumerate(flag_grid):

            clf, S = train_clf(X_train, y_train, n_cov, flag)

            acc_raw_clean = accuracy_score(y_test, clf.predict(X_test))
            print( str(flag) + "accuracy on clean data", acc_raw_clean)

           


