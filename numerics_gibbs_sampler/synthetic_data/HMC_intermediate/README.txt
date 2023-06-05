Synthetic data and teacher-student experiments using HMC sampler.

The file "percentage_convergence_intermediate.csv" contains the thermalization probabilities that were plotted in Figure 2, left panel.

The data files leading to these probabilities are not included in the supplementary material since they occupy several GigaBytes. Nonetheless we provide the code that was used to run the experiments ("MCMC_2_layer_NN_regr_torch_cluster_therm.py"), together with an example of the script file that is used to set the parameters of the MCMC. This script is named "script_example_regression_2_layer.txt".
The program autonomously generates a synthetic train and test dataset, with the specified level of noise. It should be stored in a folder called "data/inport_data"

It's possible to run HMC on this data by calling:

>>> python3 HMC_{start}_intermediate.py {delta} {num_samples} {seed}

Where start can either be "info" or "zero" for informed initialisation on the teacher weight or small norm initialisation, "delta" is the variance of the noise, "num_samples" is the number of measured samples in the simulation and "seed" is the seed of the dataset you should raead (amongst those previously generated)
The output will be saved in a folder called "data/intermediate.

The scripts "HMC_{start}_intermediate_algo.py" are for reproducing Figure 2, right panel.
