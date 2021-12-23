import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
Data operation functions for the ARA approach to Adversarial Classification
'''

def get_spam_data(path):
    '''
    Read spam data
    path -- directory where .csv is located

    return dataset and labels
    '''
    data = pd.read_csv(path)
    X = data.drop("spam", axis=1).values
    y = data.spam.values
    return X,y

def get_malware_data(path):
    '''
    Read malware data
    path -- directory where .csv is located

    return dataset and labels
    '''
    data = pd.read_csv(path)
    X = data.drop("label", axis=1).values
    y = data.label.values
    return X,y


def generate_train_test(X, y, q=0.3):
    ''''
    Generate train and test sets with test set size of q*data_size

    X -- dataset
    y -- labels
    q -- test size
    return train, test, labels_train, labels_test
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=q)
    return X_train, X_test, y_train, y_test
