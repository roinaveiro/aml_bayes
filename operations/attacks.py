import numpy as np
from samplers import *
from utils import *
from joblib import Parallel, delayed

'''
Attacks for the ARA approach to Adversarial Classification
'''

def compute_probability(x, params):
    l = params["l"]
    prb = np.zeros(l)
    for c in range(l):
        # Sample from p^*(x|x')
        sample = params["sampler_star"](x)
        # Compute p(y|x) for each x in sample
        probs = sample_label(sample, params["clf"],
            mode='evaluate', n_samples=0)[:,c]
        # Approximate with MC the value of the mean
        prb[c] = np.mean(probs, axis = 0)
    return prb

def attack_ARA(x, y, params):
    l = params["l"]
    if y < l:
        return x
    else:
        S = params["S"]
        ut_mat = params["ut_mat"]
        uts = np.expand_dims(ut_mat[:params["l"],1], axis=1)
        perturbations = original_instances_given_dist(x[S],
            n=params["distance_to_original"])

        attacks = np.ones([perturbations.shape[0], x.shape[0]], dtype=int)*x
        attacks[:,S] = perturbations
        prob_matrix = np.zeros([perturbations.shape[0], l])
        ##
        for i in range(perturbations.shape[0]): ## ESTO ES UN CHOCHO
            prob_matrix[i] = compute_probability(attacks[i], params)
        ##
        expected_ut = np.dot(prob_matrix, uts)
        idx = np.argmax(expected_ut)
        return attacks[idx]


def attack_par(i, X, y, params):
    return attack_ARA(X[i], y[i], params)

def attack_set(X, y, params):
    # num_cores=4 # it depends of the processor
    atts = Parallel(n_jobs=-1)(delayed(attack_par)(i, X, y, params) for i in range(X.shape[0]))
    return np.array(atts)


def attack_noCK(x, y, params):
    l = params["l"]
    if y < l:
        return x
    else:
        S = params["S"]
        ut_mat = params["ut_mat"]
        uts = np.expand_dims(ut_mat[:params["l"],1], axis=1)
        perturbations = original_instances_given_dist(x[S],
            n=params["distance_to_original"])

        attacks = np.ones([perturbations.shape[0], x.shape[0]], dtype=int)*x
        attacks[:,S] = perturbations
        prob_matrix = np.zeros([perturbations.shape[0], l])
        ##
        for i in range(perturbations.shape[0]): ## ESTO ES UN CHOCHO
            pr = compute_probability(attacks[i], params)
            prob_matrix[i] = np.random.uniform(pr - params["dev"]*pr,
                                                    pr + params["dev"]*pr)
        ##
        expected_ut = np.dot(prob_matrix, uts)
        idx = np.argmax(expected_ut)
        return attacks[idx]

def attack_par_noCK(i, X, y, params):
    return attack_noCK(X[i], y[i], params)

def attack_set_noCK(X, y, params):
    # num_cores=4 # it depends of the processor
    atts = Parallel(n_jobs=-1)(delayed(attack_par_noCK)(i, X, y, params) for i in range(X.shape[0]))
    return np.array(atts)



def attack_UP(x, y, params):
    l = params["l"]
    if y < l:
        return x
    else:
        S = params["S"]
        ut_mat = params["ut_mat"]
        uts = np.expand_dims(ut_mat[:params["l"],1], axis=1)
        perturbations = original_instances_given_dist(x[S],
            n=params["distance_to_original"])

        attacks = np.ones([perturbations.shape[0], x.shape[0]], dtype=int)*x
        attacks[:,S] = perturbations
        prob_matrix = np.zeros([perturbations.shape[0], l])
        ##
        for i in range(perturbations.shape[0]): ## ESTO ES UN CHOCHO
            pr = compute_probability(attacks[i], params)
            prob_matrix[i] = np.random.uniform(pr, pr + params["dev"]*pr )
        ##
        expected_ut = np.dot(prob_matrix, uts)
        idx = np.argmax(expected_ut)
        return attacks[idx]

def attack_par_UP(i, X, y, params):
    return attack_UP(X[i], y[i], params)

def attack_set_UP(X, y, params):
    # num_cores=4 # it depends on the processor
    atts = Parallel(n_jobs=-1)(delayed(attack_par_UP)(i, X, y, params) for i in range(X.shape[0]))
    return np.array(atts)




if __name__ == '__main__':

    X, y = get_spam_data("data/uciData.csv")
    X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.3)
    clf = LogisticRegression(penalty='l1', C=0.01)
    clf.fit(X,y)
    ## Get "n" more important covariates
    n=5
    weights = np.abs(clf.coef_)
    S = (-weights).argsort()[0,:n]

    params = {
                "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                "k"      : 2,     # Number of classes
                "var"    : 0.1,   # Proportion of max variance of betas
                "ut"     :  np.array([[1.0, 0.0],[0.0, 1.0]]), # Ut matrix for defender
                "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                            # what the classifier says, columns
                                            # real label!!!
                "sampler_star" : lambda x: sample_original_instance_star(x,
                 n_samples=15, rho=2, x=None, mode='sample', heuristic='uniform'),
                 ##
                 "clf" : clf,
                 "tolerance" : 3, # For ABC
                 "classes" : np.array([0,1]),
                 "S"       : S, # Set of index representing covariates with
                                             # "sufficient" information
                 "X_train"   : X_train,
                 "distance_to_original" : 2 # Numbers of changes allowed to adversary
            }


    # attack = lambda x, y: attack_ARA(x, y, params)
    print(accuracy_score(y_test, clf.predict(X_test)))

    ## Attack test set
    X_att = attack_set(X_test, y_test, params)

    print(accuracy_score(y_test, clf.predict(X_att)))
