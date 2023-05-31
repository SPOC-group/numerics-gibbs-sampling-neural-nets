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

def MLP_1_hidden_layer_w_bias(sigma,W,b_W,a,b_a,X,noise_z,noise_Phi,noise_y):
    return (sigma(X@W.T+b_W+noise_z)+noise_Phi)@(a.T)+b_a[None,:]+noise_y

def MLP_1_hidden_layer_noiseless_w_bias(sigma,W,b_W,a,b_a,X):
    return (sigma(X@W.T+b_W))@(a.T)+b_a[None,:]

def logP_unnorm_times_Delta_Z(X,W,b_W,Z,Phi,a,b_a,y,Delta_Z,Delta_Phi,Delta_y,sigma): #when this stabilizes, then it means we reached a stationary state.
    return -0.5*(torch.sum((Z-X@(W.T)-b_W[None,:])**2+((Phi-sigma(Z))**2)*(Delta_Z/Delta_Phi))+torch.sum((y-Phi@(a.T)-b_a[None,:])**2)*(Delta_Z/Delta_y))

def logP_W_1_cond_times_Delta_Z_unnorm(W_1,Cov_W_resc_XT,Cov_inv_W_resc,Z,b_1):

    """
    rescaled version of the unnormalized conditional distribution of W on X,b,Z.
    This version only implements the factor in the exponential ,i.e., -0.5(w-m_w)^T \Sigma^{-1}*Delta_Z (w-m_w)
    It exploits the fact that  \Sigma^{-1}*Delta_Z=Cov_inv_W_resc=(X.T@X)+Delta_Z*lambda_W*torch.eye(d)
    m_W is computed using the precomputed quantities
    Cov_inv_W= Cov_inv_W_resc / Delta_Z
    """
    W_minus_m_W=W_1-(Cov_W_resc_XT@(Z-b_1[None,:])).T
    return -0.5*torch.trace(W_minus_m_W@Cov_inv_W_resc@(W_minus_m_W.T))

def logP_W_l_cond_times_Delta_Z_unnorm(W_l,X_l,b_l,Z_lp1,Delta_Z_lp1,lambda_W_l):
    d_lp1=Z_lp1.shape[1]
    d_l=X_l.shape[1]
    Cov_W_inv_resc=(X_l.T)@X_l+Delta_Z_lp1*lambda_W_l*torch.eye(d_l) #Cov_inv_W= Cov_inv_W_resc / Delta_Z
    Cov_W_resc=torch.linalg.inv(Cov_W_inv_resc) #the true covariance of each row of W is Cov_W = Cov_W_resc*Delta_Z_lp1. We divide by Delta_Z_lp1 to be regularized when Delta_Z_lp1-->0 (provided that X.T@X is invertible)
    W_minus_m_W=W_l-((Cov_W_resc@(X_l.T)@(Z_lp1-b_l[None,:])).T) #mean of the weights
    return -0.5*torch.trace(W_minus_m_W@Cov_W_resc@(W_minus_m_W.T))

def therm_prob_ratio():
    pass 

def score_W_l_resc(W_l,X_l,Z_lp1,b_l,Delta_Z,lambda_W_l): 
    """
    computes the gradient (w.r.t. W) of the log posterior multiplied by Delta_Z.
    This is the score multiplied by Delta_Z
    Returns a matrix with the same shape as the weights W_l
    The true score would be score_W_l = score_W_l_resc/Delta_Zlp1
    """
    return torch.tensordot(Z_lp1-X_l@(W_l.T)-b_l[None,:], X_l,[[0],[0]])-Delta_Z*lambda_W_l*W_l
    

floating_point_precision= '32' #'32','64' #respective for using FloatTensor and DoubleTensor types
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(floating_point_precision=='32'):
    torch.set_default_tensor_type(torch.FloatTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

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
d=torch.tensor(int(input_variables[11]))
K_0=torch.tensor(int(input_variables[12]))
K=torch.tensor(int(input_variables[13]))
lambda_W_0=torch.tensor( float(input_variables[14])) 
lambda_W=torch.tensor( float(input_variables[15])) 
lambda_a_0=torch.tensor( float(input_variables[16])) 
lambda_a=torch.tensor(float(input_variables[17]))
Delta_Z_0= torch.tensor(float(input_variables[18])) 
Delta_Z= torch.tensor(float(input_variables[19]))
Delta_Phi_0=torch.tensor( float(input_variables[20])) 
Delta_Phi= torch.tensor(float(input_variables[21]))
Delta_y_0=torch.tensor( float(input_variables[22]))
Delta_y=torch.tensor( float(input_variables[23]))
teacher_activation= input_variables[24]
student_activation= input_variables[25]
max_steps=torch.tensor( int(input_variables[26]))
number_measurements=int(input_variables[27])
t_thermalization= torch.tensor(int(input_variables[28]))
lambda_b_W_0=torch.tensor(float(input_variables[29]))
lambda_b_W=torch.tensor(float(input_variables[30]))
lambda_b_a_0=torch.tensor(float(input_variables[31]))
lambda_b_a=torch.tensor(float(input_variables[32]))


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
dyn_output_file_name="dyn_th_"+input_split[:-4]+"_"+identifier+".txt"
dyn_out_file=open("./results/dyn/%s"%dyn_output_file_name, "w")
dyn_out_file.write("t test_mse train_mse W_norm a_norm y_norm_sq score_W_resc score_a_resc log_P time\n")#write header

output_file_name="MCMC_2_layer_NN_regr_teach_stud_th_%s.txt"%identifier
out_file=open("./results/%s"%output_file_name, "a")


#output_file_path="./results/%s"%output_file_name
#if(os.stat(output_file_path).st_size == 0):
#    header="num_steps final_time flag_thermalized BO_test_mse BO_predicted_test_mse last_W_test_mse last_W_train_mse avg_test_mse_therm input_file seed_teacher seed_noise_dataset seed_student seed_MCMC initialization n n_test d K_0 K lambda_W_0 lambda_W lambda_a_0 lambda_a Delta_Z_0 Delta_Z Delta_Phi_0 Delta_Phi Delta_y_0 Delta_y teacher_activation student_activation max_steps number_measurements t_thermalization \n"
#    out_file.write(header)

if(verbose):
    print("sampling dataset...\n")
#sample teacher
torch.manual_seed(seed_teacher)
W_0=torch.randn(size=[K_0,d])/torch.sqrt(lambda_W_0)
a_0=torch.randn(size=[1,K_0])/torch.sqrt(lambda_a_0)
b_W_0=torch.randn(size=[K_0])/torch.sqrt(lambda_b_W_0)#torch.zeros([K])#
b_a_0=torch.randn(size=[1])/torch.sqrt(lambda_b_a_0)#torch.zeros([1])#

#sample datasets
torch.manual_seed(seed_noise_dataset)
X=torch.randn(size=[n,d])#training data
noise_Z=torch.sqrt(Delta_Z_0)*torch.randn(size=[n,K_0])
noise_Phi=torch.sqrt(Delta_Phi_0)*torch.randn(size=[n,K_0])
noise_y=torch.sqrt(Delta_y_0)*torch.randn(size=[n,1])
                        
y=MLP_1_hidden_layer_w_bias(sigma_0,W_0,b_W_0,a_0,b_a_0,X,noise_Z,noise_Phi,noise_y) #training labels
y_noiseless=MLP_1_hidden_layer_noiseless_w_bias(sigma_0,W_0,b_W_0,a_0,b_a_0,X) #labels used to compute the training error (but not to train the network)

X_test=torch.randn(size=[n_test,d])
y_test=MLP_1_hidden_layer_noiseless_w_bias(sigma_0,W_0,b_W_0,a_0,b_a_0,X_test) 

if(verbose):
    print("initializing MCMC...")

#pick MCMC initialization
if(initialization=="zero"):
    #zero initialization (seems to work best)

    W=torch.zeros([K,d])
    a=torch.zeros([1,K])
    Z=torch.zeros([n,K])
    Phi=torch.zeros([n,K])
    b_W=torch.zeros([K])
    b_a=torch.zeros(1)

elif(initialization=="informed"):#only available if K=K_0

    Z_0=(X@W_0.T+b_W_0[None,:]+noise_Z)
    Phi_0=(sigma_0(Z_0)+noise_Phi)
    W=torch.clone(W_0)
    a=torch.clone(a_0)
    Phi=torch.clone(Phi_0)
    Z=torch.clone(Z_0)
    b_W=torch.clone(b_W_0)
    b_a=torch.clone(b_a_0)

elif(initialization=="random"):
    torch.manual_seed(seed_student)
    W=torch.randn(size=W_0.shape)/torch.sqrt(lambda_W)
    b_W=torch.randn(size=b_W.shape)/torch.sqrt(lambda_b_W)
    Z=torch.randn(size=[n,K])
    Phi=torch.randn(size=[n,K])
    a=torch.randn(size=a_0.shape)/torch.sqrt(lambda_a)
    b_a=torch.randn(size=b_a.shape)/torch.sqrt(lambda_b_a)
    
else:
    raise Exception("Invalid MCMC initialization: "+initialization)

#Precomputed things
#for sample_W_b_1_fcl (i.e. sampling W,b jointly)
sum_X=torch.sum(X,axis=0)[None,:]
up_block=torch.cat((torch.tensor([[lambda_b_W*Delta_Z+n]]),sum_X),axis=1)
down_block=torch.cat((sum_X.T,(X.T)@X+Delta_Z*lambda_W*torch.eye(d)),axis=1)
Cov_W_b_resc=torch.linalg.inv(torch.cat((up_block,down_block),axis=0))
Cholesky_Cov_W_b_resc=torch.linalg.cholesky(Cov_W_b_resc) #<---- must pass as first argument
Cov_W_b_resc_XT = Cov_W_b_resc @ torch.cat((torch.ones([n,1]),X),axis=1).T #<---- must pass this as second argument

#for sample_W_1_fcl
Cov_W_resc=torch.linalg.inv(X.T@X+Delta_Z*lambda_W*torch.eye(d))
Cholesky_Cov_W_resc=torch.linalg.cholesky(Cov_W_resc)
Cov_W_resc_XT=Cov_W_resc@(X.T)

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
y=y.to(device)
y_noiseless=y_noiseless.to(device)
y_test=y_test.to(device)
Delta_Z=Delta_Z.to(device)
Delta_Phi=Delta_Phi.to(device)
Delta_y=Delta_y.to(device)
lambda_W=lambda_W.to(device)
lambda_b_W=lambda_b_W.to(device)
lambda_a=lambda_a.to(device)
lambda_b_a=lambda_b_a.to(device)
W=W.to(device)
b_W=b_W.to(device)
Z=Z.to(device)
Phi=Phi.to(device)
a=a.to(device)
b_a=b_a.to(device)
Cholesky_Cov_W_b_resc=Cholesky_Cov_W_b_resc.to(device)
Cov_W_b_resc_XT=Cov_W_b_resc_XT.to(device)
Cholesky_Cov_W_resc=Cholesky_Cov_W_resc.to(device)
Cov_W_resc_XT=Cov_W_resc_XT.to(device)

#variables to track in the dynamics
y_pred_test_avg=torch.zeros(n_test,device=device)
y2_pred_test_avg=torch.zeros(n_test,device=device)
avg_test_mse_therm=0
flag_thermalized=0

#measure_times=[]
#measure_times=[100*i for i in range(11000)]+[1e100]
measure_times=list(torch.logspace(0,torch.log10(max_steps),number_measurements)-1.5)+[1e100]
torch.manual_seed(seed_MCMC)
start_time=time.time()
#######
#np.savez('./results/configs_K10_d50/train_set.npz',X=X.cpu().numpy(),y=y.cpu().numpy())
#np.savez('./results/dataset_K10_d50_D1e-4_st3_sd10000.npz',X_train=X.cpu().numpy(),y_train=y.cpu().numpy(),X_test=X_test.cpu().numpy(),y_test=y_test.cpu().numpy(), W_0=W_0.cpu().numpy(),b_0=b_W_0.cpu().numpy(),W_2_0=a_0.cpu().numpy(),b_2_0=b_a_0.cpu().numpy())
if(verbose):
    print("starting the MCMC...\n")
for t in range(max_steps):
    #measure stuff
    if(t>=measure_times[0]):
        measure_times.pop(0)
        y_pred_test=MLP_1_hidden_layer_noiseless_w_bias(sigma,W,b_W,a,b_a,X_test) 
        y_pred_train=MLP_1_hidden_layer_noiseless_w_bias(sigma,W,b_W,a,b_a,X)
        train_mse=torch.mean((y_noiseless-y_pred_train)**2).item()
        test_mse=torch.mean((y_test-y_pred_test)**2).item()
        W_norm=torch.linalg.norm(W).item()
        a_norm=torch.linalg.norm(a).item()
        y_norm_sq=torch.mean(y_pred_train**2).item()
        score_W_resc=torch.mean(score_W_l_resc(W,X,Z,b_W,Delta_Z,lambda_W)).item()
        score_a_resc=torch.mean(score_W_l_resc(a,Phi,y,b_a,Delta_y,lambda_a)).item()
        log_P=logP_unnorm_times_Delta_Z(X,W,b_W,Z,Phi,a,b_a,y,Delta_Z,Delta_Phi,Delta_y,sigma)
        elapsed_time=time.time()-start_time
        dyn_out_file.write("%d %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e\n"%(t, test_mse, train_mse, W_norm, a_norm, y_norm_sq, score_W_resc, score_a_resc, log_P, elapsed_time))
        dyn_out_file.flush()
        #######
        #np.savez('./results/configs_6/%s_%d_s0.npz'%(initialization[:4],t), W=W.cpu().numpy(),b_W=b_W.cpu().numpy(), Z=Z.cpu().numpy(), Phi=Phi.cpu().numpy(), a=a.cpu().numpy(), b_a=b_a.cpu().numpy())

    if(t>t_thermalization):
            flag_thermalized=1
            y_pred_test=MLP_1_hidden_layer_noiseless_w_bias(sigma,W,b_W,a,b_a,X_test) 
            y_pred_test_avg=y_pred_test_avg+y_pred_test
            y2_pred_test_avg=y2_pred_test_avg+y_pred_test**2
            avg_test_mse_therm=avg_test_mse_therm+torch.mean((y_test-y_pred_test)**2)

    #update variables
    a=mc.sample_W_l_fcl(Phi,b_a,y,lambda_a,Delta_y)
    b_a=mc.sample_b_l_fcl(a,y,Phi,Delta_y,lambda_b_a)
    Phi=mc.sample_X_L_fcl(sigma(Z),a,b_a,y,Delta_Phi,Delta_y)
    Z=mc.sample_Z_lp1_relu(X@(W.T)+b_W[None,:],Phi,Delta_Z,Delta_Phi,precise=True)
    W,b_W=mc.sample_W_b_1_fcl(Cholesky_Cov_W_b_resc,Cov_W_b_resc_XT,Z,Delta_Z)

if(verbose):
    print("MCMC finished, preparing output...")
dyn_out_file.close()
avg_test_mse_therm=(avg_test_mse_therm/(t-t_thermalization)).item()#not to be confused with the BO predictor mse.
y_avg_test=y_pred_test_avg/(t-t_thermalization)
y2_avg_test=y2_pred_test_avg/(t-t_thermalization)
y_avg_test_mse=torch.mean((y_avg_test-y_test)**2).item() #this is the BO predictor MSE (if the MCMC has thermalized)
y_avg_predicted_test_mse=torch.mean(y2_avg_test-y_avg_test**2).item()
out_file.write("%d %.1f %d %1.3e %1.3e %1.3e %1.3e %1.3e %s %d %d %d %d %s %d %d %d %d %d %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %s %s %d %d %d\n"%(t,elapsed_time, flag_thermalized, y_avg_test_mse, y_avg_predicted_test_mse,test_mse,train_mse,avg_test_mse_therm, input_split, seed_teacher, seed_noise_dataset, seed_student, seed_MCMC, initialization, n, n_test, d, K_0, K,lambda_W_0, lambda_W,lambda_a_0, lambda_a, Delta_Z_0, Delta_Z, Delta_Phi_0, Delta_Phi, Delta_y_0, Delta_y, teacher_activation, student_activation, max_steps, number_measurements, t_thermalization))
out_file.close()
if(verbose):
    print("MCMC executed successfully. Bye!")