# coding: utf-8

#this program implements an MCMC for a two layer network and trains it in the teacher student setting
#the dynamical output gets saved in ./results/dyn/
#the final output gets saved in ./results/MCMC_2_layer_NN_regr_teach_stud.txt
#the program runs mainly on the GPU 
# !!! At the moment there is only support for ReLU activation !!!
import torch
import mcmc_functions_torch as mc
import argparse
import time
import os 
import numpy as np

def MLP_class_1_hidden_layer_w_bias(sigma,W,b,W_2,b_2,X,noise_Z,noise_X_2,noise_Z_2):
    return torch.argmax((sigma(X@W.T+b+noise_Z)+noise_X_2)@(W_2.T)+b_2[None,:]+noise_Z_2, axis=1).type(torch.long)

def MLP_class_1_hidden_layer_noiseless_w_bias(sigma,W,b,W_2,b_2,X):
    return torch.argmax((sigma(X@W.T+b))@(W_2.T)+b_2[None,:],axis=1).type(torch.long)

floating_point_precision= '32' #'32','64' #respective for using FloatTensor and DoubleTensor types
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(floating_point_precision=='32'):
    tensor_type=torch.FloatTensor
else:
    tensor_type=torch.DoubleTensor
torch.set_default_tensor_type(tensor_type)

torch.set_grad_enabled(False) #globally disable the autograd engine
#parse
verbose=True
if(verbose):
    print("current device: ", device, "\n")
    print("reading inputs...\n")
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input_file_name",help="relative path to the file containing the inputs")
parser.add_argument("-id", "--identifier", default="",help="string to identify the run or group or runs")
args = parser.parse_args()
input_file_name=args.input_file_name
identifier=args.identifier
input_variables=[0]
#read inputs
with open(input_file_name,"r") as input_file:
    for line in input_file:
        split_line=line.split(' ')
        if len(split_line)>1:
            input_variables.append(split_line[1])

#assign variables
seed_teacher=int(input_variables[4])
seed_noise_dataset=int(input_variables[5])
seed_student=int(input_variables[6])
seed_MCMC=int(input_variables[7])
initialization=input_variables[8]
n=torch.tensor(int(input_variables[9])) 
n_test=torch.tensor(int(input_variables[10]))
d=torch.tensor(int(input_variables[11])) #locked at 28*28 in mnist case
K_0=torch.tensor(int(input_variables[12]))
K=torch.tensor(int(input_variables[13]))
lambda_W_0=torch.tensor( float(input_variables[14])) 
lambda_W=torch.tensor( float(input_variables[15])) 
lambda_W_2_0=torch.tensor( float(input_variables[16])) 
lambda_W_2=torch.tensor(float(input_variables[17]))
Delta_Z_0=torch.tensor(float(input_variables[18])) 
Delta_Z= torch.tensor(float(input_variables[19]))
Delta_X_2_0=torch.tensor( float(input_variables[20])) 
Delta_X_2= torch.tensor(float(input_variables[21]))
Delta_Z_2_0=torch.tensor( float(input_variables[22]))
Delta_Z_2=torch.tensor( float(input_variables[23]))
teacher_activation=input_variables[24]
student_activation=input_variables[25]
max_steps=torch.tensor( int(input_variables[26]))
number_measurements=int(input_variables[27])
t_thermalization= torch.tensor(int(input_variables[28]))
lambda_b_0=torch.tensor(float(input_variables[29]))
lambda_b=torch.tensor(float(input_variables[30]))
lambda_b_2_0=torch.tensor(float(input_variables[31]))
lambda_b_2=torch.tensor(float(input_variables[32]))
C=10 #nuber of classes

if(teacher_activation=="ReLU"):
    sigma_0=mc.ReLU
else:
    raise Exception("Invalid teacher activation function: "+teacher_activation)

if(student_activation=="ReLU"):
    sigma=mc.ReLU
#add support for absolute value and sign
else:
    raise Exception("Invalid student activation function: "+student_activation)


#simulation outputs
#For each run of the algorithm one dynamics file is created. The dynamics file is identified by the name of the script that was used to launch the simulation and the identifier.
#In addition to the dynamics file another global output file is opened in append mode. This global file only contains information regarding the whole simulation (e.g. average quantities)
#Simulations with the same identifier will be written in the same global file. Hence one should use the identifier to mark simulations that one wants to compare, or that belong to the same set of experiments.

if(verbose):
    print("initializing output files...")
os.makedirs("./results", exist_ok=True)
os.makedirs("./results/dyn",exist_ok=True)
input_split=input_file_name.split('/')[-1]
dyn_output_file_name="dyn_"+input_split[:-4]+"_"+identifier+".txt"
dyn_out_file=open("./results/dyn/%s"%dyn_output_file_name, "w")
dyn_out_file.write("t test_acc train_acc W_norm W_2_norm b_norm b_2_norm time\n")#write header

output_file_name="MCMC_2_layer_NN_regr_teach_stud_%s.txt"%identifier
out_file=open("./results/%s"%output_file_name, "a")

#output_file_path="./results/%s"%output_file_name
#if(os.stat(output_file_path).st_size == 0):
#     header= "t final_time flag_thermalized BO_test_acc test_acc train_acc input_file seed_teacher seed_noise_dataset seed_student seed_MCMC initialization n n_test d K_0 K lambda_W_0 lambda_W lambda_W_2_0 lambda_W_2 Delta_Z_0 Delta_Z Delta_X_2_0 Delta_X_2 Delta_Z_2_0 Delta_Z_2 teacher_activation student_activation max_steps number_measurements t_thermalization"
#    out_file.write(header)

if(verbose):
    print("sampling dataset...\n")


#load mnist dataset
file_mnist=np.load("./datasets/mnist.npz",allow_pickle=True)
X=torch.tensor(file_mnist["x_train"]) #use only half of the training set
X_test=torch.tensor(file_mnist['x_test']) #use only half of the test set

X=X.reshape([60000,28*28])[:n]/255
X_test=X_test.reshape([10000,28*28])[:n_test]/255
mean_X=torch.mean(X)
std_X=torch.std(X)
X=(X-mean_X)/std_X
X_test=(X_test-mean_X)/std_X

#generate teacher weights
#seed_teacher=2 (gives satistactory uniformity in label distribution)

torch.manual_seed(seed_teacher)
W_0=(torch.randn(size=[K_0,d])/torch.sqrt(lambda_W_0))
W_2_0=(torch.randn(size=[C,K_0])/torch.sqrt(lambda_W_2_0))
b_0=(torch.randn(size=[K_0])/torch.sqrt(lambda_b_0))
b_2_0=(torch.randn(size=[C])/torch.sqrt(lambda_b_2_0))

#sample noise in dataset
#seed_noise_dataset=10000
torch.manual_seed(seed_noise_dataset)
noise_Z=torch.sqrt(Delta_Z_0)*torch.randn(size=[n,K_0])
noise_X_2=torch.sqrt(Delta_X_2_0)*torch.randn(size=[n,K_0])
noise_Z_2=torch.sqrt(Delta_Z_2_0)*torch.randn(size=[n,C])

#sample datasets
#X=torch.randn(size=[n,d])
y_train=MLP_class_1_hidden_layer_w_bias(sigma_0,W_0,b_0,W_2_0,b_2_0,X,noise_Z,noise_X_2,noise_Z_2)

y_noiseless=MLP_class_1_hidden_layer_noiseless_w_bias(sigma_0,W_0,b_0,W_2_0,b_2_0,X)

Z_0=X@W_0.T+b_0[None,:]+noise_Z
X_2_0=sigma_0(Z_0)+noise_X_2
Z_2_0=X_2_0@(W_2_0.T)+b_2_0[None,:]+noise_Z_2

#generate test labels
#X_test=torch.randn(size=[n_test,d])
y_test=MLP_class_1_hidden_layer_noiseless_w_bias(sigma_0,W_0,b_0,W_2_0,b_2_0,X_test)

#here one can override the settings, to use the real MNIST labels
y_train=torch.tensor(file_mnist['y_train']).type(torch.long).flatten()[:n]
y_noiseless=y_train
y_test=torch.tensor(file_mnist['y_test']).type(torch.long).flatten()[:n_test]
#the following is to save training test set and teacher weigths
#np.savez("./results/ts_setting/red_train_set_mnist_d784_K12_D1e-3_n60k_seed_teach2_seednoise10000_rt.npz", y=y_train.cpu().numpy(),y_noiseless=y_noiseless.cpu().numpy())
#np.savez("./results/ts_setting/red_test_set_mnist_d784_K12_D1e-3_n60k_seed_teach2_rt.npz", y=y_test.cpu().numpy())
#np.savez("./results/ts_setting/teacher_weights_mnist_d784_K12_D1e-3_n60k_seed_teach2_rt.npz",W=W_0.cpu().numpy(), b=b_0.cpu().numpy(),W_2=W_2_0.cpu().numpy(),b_2=b_2_0.cpu().numpy())
####
if(verbose):
    print("initializing MCMC...")

#pick MCMC initialization
if(initialization=="zero"):
    #zero initialization (seems to work best)

    W=torch.zeros([K,d]).to(device)
    b=torch.zeros([K]).to(device)
    Z=torch.zeros([n,K]).to(device)
    X_2=torch.zeros([n,K]).to(device)
    W_2=torch.zeros([C,K]).to(device)
    b_2=torch.zeros([C]).to(device)
    Z_2=torch.zeros([n,C]).to(device)

elif(initialization=="informed"):#only available if K=K_0

    W=W_0.clone().to(device)
    b=b_0.clone().to(device)
    Z=Z_0.clone().to(device)
    X_2=X_2_0.clone().to(device)
    W_2=W_2_0.clone().to(device)
    b_2=b_2_0.clone().to(device)
    Z_2=Z_2_0.clone().to(device)

else:
    raise Exception("Invalid MCMC initialization: "+initialization)


#Precomputed things
double_type=torch.DoubleTensor

#for sample_w_1_fcl
Cov_W_resc=torch.linalg.inv(X.type(double_type).T@X.type(double_type)+Delta_Z.type(double_type)*lambda_W.type(double_type)*torch.eye(d).type(double_type))
Cholesky_Cov_W_resc=torch.linalg.cholesky(Cov_W_resc) #<--- must pass this as argument
Cov_W_resc_XT=Cov_W_resc@(X.type(double_type).T) #<--- must pass this as argument

#for sample_W_b_1_fcl (i.e. sampling W,b jointly)
n,d=X.shape
sum_X=torch.sum(X.type(double_type),axis=0)[None,:]
up_block=torch.cat((torch.tensor([[lambda_b*Delta_Z+n]]).type(double_type),sum_X),axis=1)
down_block=torch.cat((sum_X.T,(X.type(double_type).T)@X.type(double_type)+Delta_Z.type(double_type)*lambda_W.type(double_type)*torch.eye(d).type(double_type)),axis=1)
Cov_W_b_resc=torch.linalg.inv(torch.cat((up_block,down_block),axis=0))
Cholesky_Cov_W_b_resc=torch.linalg.cholesky(Cov_W_b_resc) #<---- must pass as first argument
Cov_W_b_resc_XT = Cov_W_b_resc @ torch.cat((torch.ones([n,1]).type(double_type),X.type(double_type)),axis=1).T #<---- must pass this as second argument

#cast back to float
Cholesky_Cov_W_resc=Cholesky_Cov_W_resc.type(tensor_type)
Cov_W_resc_XT=Cov_W_resc_XT.type(tensor_type)
Cholesky_Cov_W_b_resc=Cholesky_Cov_W_b_resc.type(tensor_type)
Cov_W_b_resc_XT =Cov_W_b_resc_XT.type(tensor_type)

#before this line nothing should be allocated on the GPU
if (str(device)=="cuda"):
    if(floating_point_precision=='32'):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
#after this line everything is by default allocated on the GPU

if(verbose):
    print("moving variables to %s..."%str(device))
#moving things to the GPU if possible
X=X.to(device)
X_test=X_test.to(device)
y_train=y_train.to(device)
y_noiseless=y_noiseless.to(device)
y_test=y_test.to(device)
Delta_Z=Delta_Z.to(device)
Delta_X_2=Delta_X_2.to(device)
Delta_Z_2=Delta_Z_2.to(device)
lambda_W=lambda_W.to(device)
lambda_b=lambda_b.to(device)
lambda_W_2=lambda_W_2.to(device)
lambda_b_2=lambda_b_2.to(device)
W=W.to(device)
b=b.to(device)
Z=Z.to(device)
X_2=X_2.to(device)
W_2=W_2.to(device)
b_2=b_2.to(device)
Z_2=Z_2.to(device)
Cholesky_Cov_W_b_resc=Cholesky_Cov_W_b_resc.to(device)
Cov_W_b_resc_XT=Cov_W_b_resc_XT.to(device)
Cholesky_Cov_W_resc=Cholesky_Cov_W_resc.to(device)
Cov_W_resc_XT=Cov_W_resc_XT.to(device)

#variables to track in the dynamics
y_pred_test_histo=torch.zeros([n_test,C],device=device).type(torch.long) #histogram of predicted test labels
flag_thermalized=0
measure_times=list(torch.logspace(0,torch.log10(max_steps),number_measurements)-1.5)+[0 for i in range(20)]
torch.manual_seed(seed_MCMC)
start_time=time.time()
if(verbose):
    print("starting the MCMC...\n")
for t in range(max_steps):
    #measure stuff
    if(t>=measure_times[0]):
        measure_times.pop(0)

        y_pred_train=MLP_class_1_hidden_layer_noiseless_w_bias(sigma,W,b,W_2,b_2,X)
        y_pred_test=MLP_class_1_hidden_layer_noiseless_w_bias(sigma,W,b,W_2,b_2,X_test)
        train_acc=torch.sum(y_pred_train==y_train).item()/(y_train.shape[0])
        test_acc=torch.sum(y_pred_test==y_test).item()/(y_test.shape[0])
        W_norm=torch.linalg.norm(W).item()
        W_2_norm=torch.linalg.norm(W_2).item()
        b_norm=torch.linalg.norm(b).item()
        b_2_norm=torch.linalg.norm(b_2).item()
        elapsed_time=time.time()-start_time
        dyn_out_file.write("%d %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e\n"%(t, test_acc, train_acc, W_norm, W_2_norm, b_norm, b_2_norm, elapsed_time))
        dyn_out_file.flush()

    if(t>t_thermalization): 
        flag_thermalized=1
        y_pred_test=MLP_class_1_hidden_layer_noiseless_w_bias(sigma,W,b,W_2,b_2,X_test)
        y_pred_test_histo[torch.arange(n_test),y_pred_test]+=1
            
    #update variables
 
    Z_2=mc.sample_Z_Lp1_multinomial_probit(X_2@(W_2.T)+b_2[None,:],Z_2,y_train,Delta_Z_2,precise=True)
    
    W_2=mc.sample_W_l_fcl(X_2,b_2,Z_2,lambda_W_2,Delta_Z_2)
    
    b_2=mc.sample_b_l_fcl(W_2,Z_2,X_2,Delta_Z_2,lambda_b_2)
    
    X_2=mc.sample_X_l_fcl(sigma(Z),W_2,b_2,Z_2,Delta_X_2,Delta_Z_2)
    
    Z=mc.sample_Z_lp1_relu(X@(W.T)+b[None,:],X_2,Delta_Z,Delta_X_2, precise=True)
    
    W,b=mc.sample_W_b_1_fcl(Cholesky_Cov_W_b_resc, Cov_W_b_resc_XT,Z,Delta_Z)


if(verbose):
    print("MCMC finished, preparing output...")
dyn_out_file.close()
BO_estimator_test=torch.argmax(y_pred_test_histo,axis=1)
BO_test_acc=torch.sum(BO_estimator_test==y_test).item()/(y_test.shape[0])
out_file.write("%d %.1f %d %1.4e %1.4e %1.4e %s %d %d %d %d %s %d %d %d %d %d %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %s %s %d %d %d\n"%(t,elapsed_time, flag_thermalized, BO_test_acc, test_acc, train_acc, input_split, seed_teacher, seed_noise_dataset, seed_student, seed_MCMC, initialization, n, n_test, d, K_0, K, lambda_W_0, lambda_W, lambda_W_2_0, lambda_W_2, Delta_Z_0, Delta_Z, Delta_X_2_0, Delta_X_2, Delta_Z_2_0, Delta_Z_2, teacher_activation, student_activation, max_steps, number_measurements, t_thermalization))
out_file.close() #                                                                                                                                           " %d     %.1f         %d                %1.4e     %1.4e     %1.4e      %s             %d             %d                %d            %d          %s         %d    %d    %d  %d  %d     %1.4e      %1.4e      %1.4e         %1.4e       %1.4e    %1.4e    %1.4e        %1.4e       %1.4e        %1.4e        %s                   %s              %d            %d                   %d   "
np.savez("./results/final_states/config_mnist_mcmc_%s_%s.npz"%(input_split[:-4], identifier), W=W.cpu().numpy(), b=b.cpu().numpy(), Z=Z.cpu().numpy(), X_2=X_2.cpu().numpy(), W_2=W_2.cpu().numpy(), b_2=b_2.cpu().numpy(), Z_2=Z_2.cpu().numpy(), rng_state=torch.get_rng_state().cpu().numpy())
if(verbose):
    print("MCMC executed successfully. Bye!")
