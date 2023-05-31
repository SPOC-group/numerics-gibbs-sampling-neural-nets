import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp



class sampler:
    def __init__(self, num_samples=2000, spacing=10, delta=1e-4):
        self.num_samples=num_samples
        self.spacing=spacing
        self.num_burnin_steps=0

        # Should clean up
        # self.Delta_Z_0=1e-4
        # self.Delta_Phi_0=1e-4
        # self.Delta_y_0=1e-4

        self.Delta_Z_0=delta
        self.Delta_Phi_0=delta
        self.Delta_y_0=delta
        
        self.d=50
        self.K=10
        self.lambda_a_0=self.K
        self.lambda_W_0=self.d
        self.lambda_W=self.lambda_W_0
        self.lambda_a=self.lambda_a_0
        self.n=2040
        self.n_test=2000
        self.seed_teacher=3
        self.seed_noise_dataset=10000
        self.seed_student=20000
        self.seed_mcmc=30000
        self.sigma_0=tf.nn.relu
        self.sigma=self.sigma_0

        self.Delta_Z=self.Delta_Z_0
        self.Delta_Phi=self.Delta_Phi_0
        self.Delta_y=self.Delta_y_0

        self.samples_result = None

    @tf.function
    def logP_unnorm(self, *current_state):
        #current_state=(W,b_W, Z, Phi, a,b_a)
        W=current_state[0]
        b_W=current_state[1]
        Z=current_state[2]
        Phi=current_state[3]
        a=current_state[4]
        b_a=current_state[5]
        logP=-0.5*(tf.reduce_sum((Z-self.X@tf.transpose(W) - b_W)**2/self.Delta_Z+((Phi-self.sigma(Z))**2)/self.Delta_Phi)+tf.reduce_sum((self.y-Phi@tf.transpose(a) - b_a)**2)/self.Delta_y) - tf.norm(W)**2*self.lambda_W_0/2 - tf.norm(a)**2*self.lambda_a_0/2 - tf.norm(b_W)**2*self.lambda_W_0/2- tf.norm(b_a)**2*self.lambda_a_0/2 
        #print(logP)
        return logP


    # @tf.function
    def MLP_1_hidden_layer(self, sigma,W,b_W,a,b_a,X,noise_Z,noise_Phi,noise_y):
        return (sigma(X@tf.transpose(W)+b_W+noise_Z))@(tf.transpose(a))+b_a+noise_y


    # @tf.function
    def MLP_1_hidden_layer_noiseless(self, sigma,W,b_W,a,b_a,X):
        return (sigma(X@tf.transpose(W)+b_W))@(tf.transpose(a))+b_a


    # @tf.function
    def make_data(self):
        #sample teacher
        tf.random.set_seed(self.seed_teacher)
        self.W_0=tf.random.normal([self.K,self.d])/np.sqrt(self.lambda_W_0)
        self.a_0=tf.random.normal([1,self.K])/np.sqrt(self.lambda_a_0)

        #sample dataset
        tf.random.set_seed(self.seed_teacher)
        self.X=tf.random.normal([self.n,self.d])
        noise_Z=tf.random.normal([self.n,self.K])*tf.math.sqrt(self.Delta_Z_0)
        noise_Phi=tf.random.normal([self.n,self.K])*tf.math.sqrt(self.Delta_Phi_0)
        noise_y=noise_Z=tf.random.normal([self.n,1])*tf.math.sqrt(self.Delta_y_0)

        self.y=self.MLP_1_hidden_layer(self.sigma,self.W_0,self.a_0,self.X,noise_Z,noise_Phi,noise_y)
        self.y_noiseless=self.MLP_1_hidden_layer_noiseless(self.sigma,self.W_0,self.a_0,self.X)
        self.Z_0=self.X@tf.transpose(self.W_0)
        self.Phi_0=self.sigma(self.Z_0)

        #test set
        self.X_test=tf.random.normal([self.n_test,self.d])
        self.y_test=self.MLP_1_hidden_layer_noiseless(self.sigma,self.W_0,self.a_0,self.X_test)


    def read_data(self, file):
        data = np.load(file)

        self.X = data["X_train"]
        self.y = data["y_train_noiseless"]

        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

        self.W_0 = data["W"]
        self.b_W_0 = data["b_W"]
        self.a_0 = data["a"]
        self.b_a_0 = data["b_a"]
        self.Z_0 = self.X@np.transpose(self.W_0) + self.b_W_0 #+ np.random.normal(0,1e-5,size=self.b_W_0.shape)
        self.Phi_0 = self.sigma(self.Z_0)

        self.n, self.d = self.X.shape
        self.n_test, self.d = self.X_test.shape
        self.K = self.b_W_0.shape[0]

    
    def initialise_weights(self, initialisation):
        if initialisation not in ["Zero", "Informed", "Random"]:
            raise ValueError('Initialisation can only be "Zero", "Informed" or "Random"')
        self.initialisation = initialisation

        if self.initialisation == "Informed":
            W=tf.identity(self.W_0)
            Z=tf.identity(self.Z_0)
            Phi=tf.identity(self.Phi_0)
            a=tf.identity(self.a_0)
            b_W=tf.identity(self.b_W_0)
            b_a=tf.identity(self.b_a_0)

        if self.initialisation == "Zero":
            # W=tf.zeros([self.K,self.d])
            # a=tf.zeros([1,self.K])
            # Phi=tf.zeros([self.n,self.K])
            # Z=tf.zeros([self.n,self.K])
            # b_W=tf.zeros([1,self.K])
            # b_a=tf.zeros([1])

            tf.random.set_seed(self.seed_student)
            W=tf.random.normal([self.K,self.d], 0, 1e-4)/np.sqrt(self.lambda_W_0)
            a=tf.random.normal([1,self.K], 0, 1e-4)/np.sqrt(self.lambda_a_0)
            Phi=tf.random.normal([self.n,self.K], 0, 1e-4)
            Z=tf.random.normal([self.n,self.K], 0, 1e-4)
            b_W=tf.random.normal([1,self.K], 0, 1e-4)/np.sqrt(self.lambda_W_0)
            b_a=tf.random.normal([1], 0, 1e-4)/np.sqrt(self.lambda_a_0)


        if self.initialisation == "Random":
            tf.random.set_seed(self.seed_student)
            W=tf.random.normal([self.K,self.d])/np.sqrt(self.lambda_W_0)
            a=tf.random.normal([1,self.K])/np.sqrt(self.lambda_a_0)
            Phi=tf.random.normal([self.n,self.K])
            Z=tf.random.normal([self.n,self.K])
            b_W=tf.random.normal([1,self.K])
            b_a=tf.random.normal([1])

        self.initial_state=(W,b_W, Z, Phi, a,b_a)


    # @tf.function
    def gaussian_proposal(self, *current_state_seed):
        current_state=current_state_seed[0]
        #print(current_state_seed[1])
        sigma=1e-5 #gaussian proposal with this standard deviation. Use sigma=1e-5 for informed initialization, 3
        proposed_state=()
        for var in current_state:
            proposed_state=proposed_state + (var+tf.random.normal(var.shape)*sigma,)
        return proposed_state


    # @tf.function
    def make_kernel(self, kernel_choice, params):
        if kernel_choice not in ["HMC", "AdaHMC", "MetropolisLangevin", "NoUTs", "RWMetropolis", "Langevin"]:
            raise ValueError('Kernel can only be "HMC", "AdaHMC", "MetropolisLangevin", "NoUTs", "RWMetropolis", "Langevin"')
        self.kernel_choice = kernel_choice

        #Different MCMC kernels

        num_leapfrog_steps = params["num_leapfrog_steps"]
        step_size = params["step_size"]

        vanilla_hmc=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.logP_unnorm,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size) #for informed, zero, random initializations use step_size=5e-5 is fine. For zero and random initialization use step_size=1e-5
        #seems to thermalize at least from the informed initialization. From the zero one it instead the test error first increases and then starts decreasing. From the random initialization it decreases monotonically ut seems to get tuck on a high palteau aroung test_mse=0.75


        #this one is supposed to change adaptively the step size.
        step_adaptive_hmc=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.logP_unnorm,
                num_leapfrog_steps=200,
                step_size=0.01),
            num_adaptation_steps=int(self.num_burnin_steps * 0.8))


        #did not manage to obtain a sample from this, no matter how small I pick the step size. Gives a misrterious error after running for some time without producing any sample
        metropolis_langevin=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
                target_log_prob_fn=self.logP_unnorm,
                step_size=step_size)

        nuts=tfp.mcmc.NoUTurnSampler( #extremely slow for some reason 
            target_log_prob_fn=self.logP_unnorm,
            step_size=1e-6,
            max_tree_depth=10,
            max_energy_diff=1000.0,
            unrolled_leapfrog_steps=1,
            parallel_iterations=10,
            experimental_shard_axis_names=None,
            name=None
        )

        rw_metropolis=tfp.mcmc.RandomWalkMetropolis( #with 'new_state_fn=gaussian_proposal' one should obtain 
            target_log_prob_fn=self.logP_unnorm,
            new_state_fn=self.gaussian_proposal,
            experimental_shard_axis_names=None,
            name=None
        )

        #did not manage to make this run: it gives a misterious error
        uncalibrated_langevin=tfp.mcmc.UncalibratedLangevin(
            target_log_prob_fn=self.logP_unnorm,
            step_size=0.001,
            volatility_fn=None,
            parallel_iterations=10,
            compute_acceptance=True,
            experimental_shard_axis_names=None,
            name=None
        )

        #pick kernel here

        if kernel_choice == "HMC":
            self.kernel = vanilla_hmc

        if kernel_choice == "AdaHMC":
            self.kernel = step_adaptive_hmc

        if kernel_choice == "MetropolisLangevin":
            self.kernel = metropolis_langevin
        
        if kernel_choice == "NoUTs":
            self.kernel = nuts

        if kernel_choice == "RWMetropolis":
            self.kernel = rw_metropolis

        if kernel_choice == "Langevin":
            self.kernel = uncalibrated_langevin


    # @tf.function(experimental_compile=True)
    def _sample(self, num_samples, num_burnin_steps, initial_state, kernel):
        samples = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=self.spacing,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=None) 
        return samples

    def sample(self):
        self.samples_result = self._sample(self.num_samples, self.num_burnin_steps, self.initial_state, self.kernel)


    # @tf.function
    def make_observables(self):
        if self.samples_result==None:
            raise MemoryError("You should run the MCMC first!")

        train_mse=[]
        test_mse=[]
        W_norm=[]
        a_norm=[]
        times=[]
        for t in range(self.num_samples):
            W=self.samples_result[0][t]
            b_W=self.samples_result[1][t]
            a=self.samples_result[4][t]
            b_a=self.samples_result[5][t]

            y_pred_train = self.MLP_1_hidden_layer_noiseless(self.sigma,W,b_W,a,b_a,self.X)
            y_pred_test  = self.MLP_1_hidden_layer_noiseless(self.sigma,W,b_W,a,b_a,self.X_test)

            train_mse.append(tf.math.reduce_mean((y_pred_train-self.y)**2).numpy())
            test_mse.append(tf.math.reduce_mean((y_pred_test-self.y_test)**2).numpy())
            W_norm.append(tf.reduce_sum(W**2).numpy())
            a_norm.append(tf.reduce_sum(a**2).numpy())
            times.append(self.spacing*t + self.num_burnin_steps)

        self.train_mse = train_mse
        self.test_mse = test_mse
        self.W_norm = W_norm
        self.a_norm = a_norm
        self.times = times

        return times, train_mse