# Code for the paper: Adversarial Machine Learning: Bayesian Perspectives

This repository contains code for reproducing the experiments in the **Adversarial Machine Learning: Bayesian Perspectives** paper.

## Protecting during operations

The environment containing all relevant libraries for this batch of experiments is `acra2.yml`.
The `operations` folder contains the code corresponding to the ML robustification approach during operations. 
The main files are the following:

* `data.py`:  contains functions to load different data sets and a function to generate train and test sets.

* `models.py`: contains sklearn implementations of the different base classification models used.

* `attacks.py`: contains functions to implement different types of attacks. 

* `samplers.py`: contains the main samplers used in ABC. In particular, contains functions to sample from utilities, probabilities, original instance and original instance given observed one.

* `inference.py`: contains functions to perform adversarial robust inference on a new instance class given its covariates.

* `utils.py`: contains some auxiliar functions.

### Reproducibility Workflow

Follow the next steps to reproduce the results in Tables 1 and 2 of the paper:

1. Install the environment containing all dependencies
`conda env create -f acra2.yml`

2. Activate environment
`conda activate acra2`

3. Run 
`python experiments_classifier.py`
This creates the results needed to compare performance of different classifiers on tainted data, with and without protection (Table 1, first 4 columns).
Results are stored in `results/spam/multiple_classifiers`

4. Run 
`python experiments_noCK.py`
This creates the results needed to comparecommon knowledge protection versus the proposed Bayesian protection under different baseline classifiers. (Table 2).
Results are stored in `results/spam/high_low_var_all/`

5. The `acra_spam_results.Rmd` R Markdown can be executed to process the generated results and exactly reproduce Tables 1 and 2.
The following files can be used to generate additional results (not present in the paper):

* `experiments_nsamples.py`: to study how different number of samples used in the MC approximation of the robbust adversarial posterior predictive utilities affect performance.

* `experiments_tolerance.py`: to study how the tolerance parameter in ABC affects performance.


## Protecting during training

The `training` folder contains the code corresponding to the ML robustification approach during training. The main dependency is the PyTorch library to define and train the different neural architectures. The main scripts are the following:

* `models.py`: defines several network architectures in Pytorch, ready to be used with the experiments. Those are a Multi-Layered Perceptron (MLP), a simple Convolutional Network, and a ResNet.

* `optimizers.py`: implements the optimizers and attack routines (fast gradient sign method, with aditional multiple steps, PGD).

* `utils_train.py`: defines the dataloaders, several functions to train with the different robustification approaches, and evaluation functions.

* `adversarial_examples_AT.ipynb`: the notebook used to run the experiments and plot the graphs from the paper.


### Reproducibility Workflow

Follow these steps to reproduce the Figure 2 of our paper:

1. Execute the first part of the notebook `adversarial_examples_AT.ipynb`. The network hyperparameters and architecture are defined there.

2. Execute the following sections (called "Training with ...") of the notebook to train the network with any of the robustification approaches. Those will save the updated weights in different files for each training method.

3. Execute the section "Full atack & defense evaluation" to attack the validation set and evaluate the according previous models over it. The final cell will plot the graph for a configuration of the variables. If you want to try the other dataset, replace `KMNIST` with `FMNIST` and repeat the execution of the notebook.

And follow these steps to reproduce the last column of Table 1:

1. Configure the environment with the Reproducibility workflow from Protecting during operations. Copy the scripts `training/table1/single_exp.py` and `training/table1/models.py` to `operations/`. These will copy the script for the robust training approach of the appropiate models. 

2. Run `python operations/single_exp.py` to perform the experiments. A csv will be created for each experiment, after that you can average results etc.
