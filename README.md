# Code for the paper: Adversarial Machine Learning: Bayesian Perspectives

This repository contains code for reproducing the experiments in the **Adversarial Machine Learning: Bayesian Perspectives** paper.

## Protecting during operations

The `operations` folder contains the code corresponding to the ML robustification approach during operations. The main files are the following:

* `data.py`:  contains functions to load different data sets and a function to generate train and test sets.

* `models.py`: contains sklearn implementations of the different base classification models used.

* `attacks.py`: contains functions to implement different types of attacks. 

* `samplers.py`: contains the main samplers used in ABC. In particular, contains functions to sample from utilities, probabilities, original instance and original instance given observed one.

* `inference.py`: contains functions to perform adversarial robust inference on a new instance class given its covariates.

* `utils.py`: contains some auxiliar functions.

To reproduce the experiments in the papers, the next functions can be used. Note that the dataset used must
be specified modifyng the line in which X and y are defined. For instance, to load the spam dataset, write
`X, y = get_spam_data("data/clean_imdb_sent_2.csv")`:

* `experiments_classifier.py`: to compare performance of different classifiers on tainted data, with and without protection.

* `experiments_noCK.py`: to compare common knowledge protection versus the proposed Bayesian protection under different baseline classifiers.

* `experiments_nsamples.py`: to study how different number of samples to approximate posterior predictive utilities affect performance.

* `experiments_tolerance.py`: to study how the tolerance parameter in ABC affects performance.



## Protecting during training

The `training` folder contains the code corresponding to the ML robustification approach during training
