import numpy as np
import pandas as pd
from data import *
from utils import *
from models import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''
Relevant samplers for the ARA approach to Adversarial Classification
'''

def sample_original_instance(x_mod, n_samples, params):
    '''
    ABC function to sample from p(x|x')
    '''
    S = params["S"]
    X_train = params["X_train"]
    clf = params["clf"]
    tolerance = params["tolerance"]
    samples = np.zeros([n_samples, x_mod.shape[0]], dtype=int)
    for i in range(n_samples): ## Parallelize
        dist = params["tolerance"] + 1 # Condition to enter in the while loop
        x_final = sample_instance(X_train)[0]
        ##
        if params["stop"]:
            for j in range(params["max_iter"]):
                x = sample_instance(X_train)[0] ## Watch out! Dimensions
                probs = sample_label(x, clf)[0]
                y = np.random.choice(params["classes"], p=probs)
                x_tilde = sample_transformed_instance(x, y, params)
                dist_tilde = distance(x_tilde[S], x_mod[S])
                if dist_tilde <= tolerance:
                    dist = dist_tilde
                    x_final = x
                    break
                elif dist_tilde < dist:
                    x_final = x
                    dist = dist_tilde
            samples[i] = x_final

        else:
            while dist > tolerance:
                x = sample_instance(X_train)[0] ## Watch out! Dimensions
                probs = sample_label(x, clf)[0]
                y = np.random.choice(params["classes"], p=probs)
                x_tilde = sample_transformed_instance(x, y, params)
                dist = distance(x_tilde[S], x_mod[S])
            samples[i] = x
    return samples

def sample_instance(X_train, n_samples=1):
    '''
    Get n_samples from p(x)
    Easy version: get samples from training set
    '''
    idx = np.random.choice(range(X_train.shape[0]), size=n_samples)
    return X_train[idx]



def sample_transformed_instance(x, y, params):
    '''
    For ABC, sample just one instance of p(x'|x,y)
    * Good labels are indexed as 0,1,...,l-1
    * If mode is "sample", a sample is obtained
    * If mode is "evaluate", probability is computed and returned
    '''
    l = params["l"]
    if y < l:
        return x
    else:
        S = params["S"]
        uts = sample_utility(y, params)
        perturbations = original_instances_given_dist(x[S], n=params["distance_to_original"])
        attacks = np.ones([perturbations.shape[0], x.shape[0]], dtype=int)*x
        attacks[:,S] = perturbations
        prob_matrix = np.zeros([perturbations.shape[0], l])
        ##
        for i in range(perturbations.shape[0]): ## ESTO ES UN CHOCHO
            prob_matrix[i] = sample_probability(attacks[i], params)
        ##
        expected_ut = np.dot(prob_matrix, uts)
        idx = np.argmax(expected_ut)
        return attacks[idx]


def sample_label(X, clf, mode='evaluate', n_samples=0):
    '''
    Sample or evaluate p(y|x)
    * If mode is 'sample', a sample is obtained
    * If mode is 'evaluate', probability is computed and returned
    X -- dataset (ndarray)
    clf -- classifier (obj)
    n_samples -- number of samples to get
    '''
    if X.ndim == 1:
        X = np.expand_dims(X, 0)
        
    if mode == 'evaluate':
        return clf.predict_proba(X)

    if mode == 'sample':
        pass


def sample_utility(i, params):
    '''
    Sample a utility for ARA

    * c -- label predicted by classifier
    * i -- real label of instance
    * attacker_ut -- utility matrix with samplers
    * n_samples -- number of samples to get

    '''
    l = params["l"]
    assert i >= l,  "Watchout class is good"
    ut_mat = params["ut_mat"]
    ut_samples = np.zeros([params["l"],1])
    var = params["var"]
    for j in range(l):
        if var == 0:
            ut_samples[j] = ut_mat[j, i]
        else:
            ut_samples[j] = ut_mat[j, i]
            #alpha, beta = get_beta_parameters(ut_mat[j, i], var)
            #ut_samples[j] = np.random.beta(alpha, beta)

    return ut_samples

def sample_probability(x, params):
    '''
    Sample a probability for ARA. DOUBLE-CHECK
    '''
    l = params["l"]
    var = params["var"]
    prob_samples = np.zeros(params["l"])
    for c in range(l):
        # Sample from p^*(x|x')
        sample = params["sampler_star"](x)
        # Compute p(y|x) for each x in sample
        probs = sample_label(sample, params["clf"],
            mode='evaluate', n_samples=0)[:,c]
        # Approximate with MC the value of the mean
        mean = np.mean(probs, axis = 0)

        if var==0:
            # CK, return mean of beta
            prob_samples[c] = mean
        else:
            # Get parameters of beta distribution
            alpha, beta = get_beta_parameters(mean, var)
            prob_samples[c] = np.random.beta(alpha, beta)

    return prob_samples

def sample_original_instance_star(x_mod, n_samples, rho, x=None, mode='sample', heuristic='uniform'):
    '''
    Sample or evaluate p(x|x') using a metric based approach.

    * x_mod -- original instance
    * rho -- maximum allowed distance

    '''
    N = x_mod.shape[0]
    if heuristic == 'uniform':
        ##
        pr, tot = get_bin_probs(N, rho)
        if mode == 'sample':
            lengths = np.random.choice( range(rho+1), p=pr, size=n_samples )
            indices = [np.random.choice( range(N), replace = False, size = rho) for rho in lengths]
            samples = np.stack( [change_instance(x_mod, set) for set in indices], axis=0 )
            return samples

        if mode == 'evaluate':
            if distance(x, x_mod) <= rho:
                return 1/tot
            else:
                return 0

        if heuristic == 'penalized_distance':
            pass


        #return

# Some tests

if __name__ == '__main__':

    X, y = get_spam_data("data/uciData.csv")
    clf = LogisticRegression()
    X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.3)
    clf.fit(X_train, y_train)


    params = {
                "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                "k"      : 2,     # Number of classes
                "var"    : 0.1,   # Proportion of max variance of betas
                "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                            # what the classifier says, columns
                                            # real label!!!
                "sampler_star" : lambda x: sample_original_instance_star(x,
                 n_samples=15, rho=1, x=None, mode='sample', heuristic='uniform'),
                 "clf" : clf,
                 "tolerance" : 1, # For ABC
                 "classes" : np.array([0,1]),
                 "S"       : np.array([1,3]), # Set of index representing covariates with
                                             # "sufficient" information
                 "X_train"   : X_train
            }


    if False:
        x = X[0]
        samples = sample_original_instance_star(x, n_samples=10, rho=10)
        ss = sample_label(samples, clf)
        print( np.mean(ss, axis=0)  )

        sampler = lambda x, n: sample_original_instance_star(x, n, rho=10)
        print( sample_probability_c(x, 1, sampler, 10, 100, 0.9) )

    ##
