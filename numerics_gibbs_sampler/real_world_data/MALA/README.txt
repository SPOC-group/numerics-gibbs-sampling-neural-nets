Real-world data using MAtropolis Adjusted Langevin Algorithm (MALA).

The program used to run MALA is a custom code built on top of pytorch. The motivation behind this was that other existing implementations of MALA don't run on the GPU, and were found to be too slow in the case of MNIST. Our implementations for MLP and CNN are respectively contained in "MALA_mlp_mnist_cluster.py" and "MALA_cnn_mnist_cluster.py".
To run the programs execute
>>> python3 ./MALA_mlp_mnist_cluster.py -Delta 0.001 -lr 1e-5 -id test_experiment -nmeas 1000

This executes the program using Delta 0.001 learning rate 1e-5, experiment name "test_experiment" and 
taking 1000 measurements logarithmically spaced along the dynamics.

The program will throw an error unless the MNIST dataset (in the form of a file "mnist.npz", downloadable from https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy )is found in the directory "./datasets/mnist.npz".

The curves plotted in figure 3 are reported in the files
"dyn_MALA_mlp_mnist_D2.0e+00_lr2.0e-06_mlp_runs.txt" and "dyn_MALA_cnn_mnist_D1.0e+01_lr5.0e-06_cnn_runs" respectively for MLP, CNN.












