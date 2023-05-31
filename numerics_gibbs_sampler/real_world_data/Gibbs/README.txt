Real-world data using Gibbs sampler on intermediate noise posterior.
The programs for the MLP and CNN are respectively "MCMC_2_layer_NN_class_torch_cluster_MNIST.py" and  "MCMC_CNN_class_torch_cluster_MNIST.py".
We include as the respective scripts to run the programs in the files "cnn_Xmnist_ymnist_zero_st20000_smc30000_D1e2_300k" and "mlp_Xmnist_ymnist_zero_n60k_D2e0_300k.txt"
These scripts have the parameters that reproduce the data in Figure 3.

The program is run with 

>>> python3 ./MCMC_CNN_class_torch_cluster_MNIST.py -in cnn_Xmnist_ymnist_zero_st20000_smc30000_D1e2_300k -id name_experiment

The program will throw an error unless the MNIST dataset (in the form of a file "mnist.npz", downloadable from https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy )is found in the directory "./datasets/mnist.npz".


The data shown in Figure 3 is contained in the following files: "dyn_mlp_Xmnist_ymnist_zero_n60k_D2e0_300k_test_deltas_mnist.txt" for the MLP and "dyn_cnn_Xmnist_ymnist_zero_st20000_smc30000_D1e2_300k_cnn_adaminfo.txt" for the CNN.














