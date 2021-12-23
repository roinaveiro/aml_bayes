import numpy as np
import scipy.special
from data import *

def get_bin_probs(N, rho):
    binoms = np.array( [scipy.special.binom(N, k) for k in range(rho + 1)] )
    return  binoms / np.sum(binoms), np.sum(binoms)

def change_instance(x, set):
    '''
    Change instance flipping the values indicated by the set of indices
    '''
    x_mod = np.copy(x)
    x_mod[set] = np.abs( x_mod[set] - 1 )
    return x_mod

def distance(x, x_mod): # Just for the Bag-Of-Words representation!
    return np.sum( np.abs(x - x_mod) )

def get_beta_parameters(mu, k):
    '''
    * mu -- mean of the beta distribution
    * k -- proportion of maximum variance to have convex beta
    '''
    var = k * mu * min(mu * (1.0 - mu) / (1.0 + mu) , (1.0 - mu)**2 / (2.0 - mu))
    alpha = ( (1-mu) / var - 1 / mu ) * mu**2
    beta = alpha * ( 1/mu - 1 )

    return alpha, beta

def original_instances_given_dist(X, n):
    """
    For a given email $X$, this function computes $\mathcal{A}(X)$
    under some attack strategy (n word transformation, in this particular case).
    For each $a \in \mathcal{A}(X)$, it computes $a(X)$, and returns an array
    containing all $a(X)$.
    """
    def add1_rem1(X):
        X = np.reshape(X, (1,-1))
        t1 = np.logical_or(X, np.identity(X.shape[1])).astype(int)
        X = np.logical_not(X).astype(int)
        t2 = np.logical_or(X, np.identity(X.shape[1]))
        t2 = np.logical_not(t2).astype(int)
        z = np.concatenate( (t1, t2), axis=0)
        return( np.unique(z, axis=0) )


    X = np.reshape(X, (1,-1))
    z = np.apply_along_axis(add1_rem1, 1, X)
    z = z.reshape(z.shape[0]*z.shape[1], z.shape[2])
    for i in range(1,n):
        z = np.apply_along_axis(add1_rem1, 1, z)
        z = z.reshape(z.shape[0]*z.shape[1], z.shape[2])

    return(np.unique(np.insert(z, 0, X, 0), axis = 0))

if __name__ == '__main__':
    X, y = get_spam_data("data/uciData.csv")
    x = X[0]
    z = original_instances_given_dist(x, n=3)
    print(z)







    #print("...................................................................")
    #ax = getxax(x, n=1)
    #print(ax[2])
    #print("...................................................................")
    #print(len(ax))
