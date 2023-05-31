Synthetic data and teacher-student experiments using the Gibbs sampler.

The file "thermalization_probability_noiseless_gibbs.txt" contains the thermalization probabilities that were plotten in Figure 2, left panel. In particular one plots "thermalization_probability" on the y axis and "avg_equilibrium_test_mse_info" on the  axis.

The data files leading to these probabilities are not included in the supplementary material since they occupy several GigaBytes. Nonetheless we provide the code that was used to run the experiments ("MCMC_2_layer_NN_regr_torch_cluster_therm.py"), together with an example of the script file that is used to set the parameters of the MCMC. This script is named "script_example_regression_2_layer.txt".
The program autonomously generates a synthetic train and test dataset, with the specified level of noise. Then it runs the Gibbs sampler on it.
During the execution it saves the dynamics of, among other quantities, training and test MSE, score (U) function for the weights, as defined in the paper.
To run this program it is necessary to have the Gibbs sampler's functions (named "mcmc_functions_torch.py") in the same folder.
To run use the command 

>>> python3 ./MCMC_2_layer_NN_regr_torch_cluster_therm.py -in script_example_regression_2_layer.txt -id experiment_name


A modified version of "MCMC_2_layer_NN_regr_torch_cluster_therm.py" is used to compute the multiple chain, or R hat, statistic. The modification simply involves saving the weigths of the network every 100 Gibbs steps. Then a program is used to read these files and compute the R hat statistic. The files containing the weights at various times are not included for lack of space.
Nonetheless we include the output of the R hat statistic, which is used to produce figure 1, 4, in the files "R_hat_quantiles_zero_info_every_100_fig1_fig4.npz" and "R_hat_quantiles_rand2_rand3_every_100_fig1_fig4.npz".
The other lines in figure 1 were computed using the data in "dyn_th_comp_conv_exp_50MLP_10_info_2M_seed0_multi_init.txt", "dyn_th_comp_conv_exp_50MLP_10_zero_2M_seed0_multi_init", "dyn_th_rand_prior_d50_K10_seedstud2_rand_prior_wonly.txt", dyn_th_rand_prior_d50_K10_seedstud3_rand_prior_wonly.txt" respectively for the informed, zero, and two random initializations. 

Instead the Gibbs runs shown in the right panel of Figure 2 come from the data in "dyn_gibbs_noiseless_dset_s10_zero_D4.6e-04_nl_mD_fig2.txt" and "dyn_gibbs_noiseless_dset_s10_info_D4.6e-04_nl_mD_fig2.txt" respectively for the zero and informed initializations.







