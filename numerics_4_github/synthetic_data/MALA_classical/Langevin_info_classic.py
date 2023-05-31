from HMC_sampler_classic import sampler
from sys import argv
import numpy as np

def main():
    algo = "MetropolisLangevin"
    init = "Informed"
    delta = float(argv[1])
    num_samples = int(argv[2])

    params = {"num_leapfrog_steps":20, "step_size":float(argv[5])}
    sampler_HMC = sampler(num_samples=num_samples, spacing=int(argv[4]), delta=delta)
    
    id = int(argv[3])

    np.random.seed(200+id)
    file = f"data/inport_data/mlp_d50_K10_DeltaZ1.0e-03_DeltaPhi1.0e-03_Deltay1.0e-03_n2084_seedteach{id}_seednoise{10000+id}.npz"

    sampler_HMC.read_data(file)
    # sampler_HMC.make_data()

    sampler_HMC.initialise_weights(initialisation=init)
    sampler_HMC.make_kernel(kernel_choice=algo, params=params)
    sampler_HMC.sample()
    sampler_HMC.make_observables()

    time = sampler_HMC.times
    train_mse = np.array(sampler_HMC.train_mse)
    test_mse = np.array(sampler_HMC.test_mse)


    np.savetxt(f"data/classic/{algo}_{init}_delta_{delta}_train_{id}.csv", train_mse, delimiter=",")
    np.savetxt(f"data/classic/{algo}_{init}_delta_{delta}_test_{id}.csv", test_mse, delimiter=",")


    return



if __name__=="__main__":
    main()