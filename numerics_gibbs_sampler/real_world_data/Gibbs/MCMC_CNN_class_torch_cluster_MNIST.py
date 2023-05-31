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
def CNN_1hl(X,W,b,s_y,s_x,H_P,W_P,W_2,b_2,noise_Z,noise_Z_2,noise_X_2,noise_Z_3):
    Z=mc.conv2d_layer(X,W,s_y,s_x)+b[None,:,None,None]+noise_Z
    Z_2=mc.average_pool2d(Z,H_P,W_P)+noise_Z_2
    X_2=mc.ReLU(Z_2).reshape([Z_2.shape[0],-1])+noise_X_2
    Z_3=X_2@(W_2.T)+b_2[None,:]+noise_Z_3
    return torch.argmax(Z_3,axis=1).type(torch.long)

def CNN_1hl_noiseless(X,W,b,s_y,s_x,H_P,W_P,W_2,b_2):
    Z=mc.conv2d_layer(X,W,s_y,s_x)+b[None,:,None,None]
    Z_2=mc.average_pool2d(Z,H_P,W_P)
    X_2=mc.ReLU(Z_2).reshape([Z_2.shape[0],-1])
    Z_3=X_2@(W_2.T)+b_2[None,:]
    return torch.argmax(Z_3,axis=1).type(torch.long)


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
max_steps=torch.tensor(int(input_variables[8]))
number_measurements=int(input_variables[9])
t_thermalization= torch.tensor(int(input_variables[10]))
initialization=input_variables[11]
n=torch.tensor(int(input_variables[12])) #locked at 28*28 in mnist case
n_test=torch.tensor(int(input_variables[13]))
H_X=torch.tensor(int(input_variables[14]))
W_X=torch.tensor(int(input_variables[15]))
C_X=torch.tensor(int(input_variables[16]))
C_Z=torch.tensor(int(input_variables[17]))
H_W=torch.tensor(int(input_variables[18]))
W_W=torch.tensor(int(input_variables[19]))
s_y=torch.tensor(int(input_variables[20]))
s_x=torch.tensor(int(input_variables[21]))
H_P=torch.tensor(int(input_variables[22]))
W_P=torch.tensor(int(input_variables[23]))
C=torch.tensor(int(input_variables[24]))
Delta=torch.tensor(float(input_variables[25])) 
lambda_W=torch.tensor(float(input_variables[26])) 
lambda_W_2=torch.tensor(float(input_variables[27]))
lambda_b=torch.tensor(float(input_variables[28]))
lambda_b_2=torch.tensor(float(input_variables[29]))
inputs=input_variables[30]
labels=input_variables[31]

H_Z=torch.div((H_X-H_W),s_y,rounding_mode='floor')+1
W_Z=torch.div((W_X-W_W),s_x,rounding_mode='floor')+1
H_Z_2=torch.div(H_Z,H_P,rounding_mode='floor')
W_Z_2=torch.div(W_Z,W_P,rounding_mode='floor')
d_W_2=C_Z*H_Z_2*W_Z_2

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
dyn_out_file.write("t test_acc train_acc W_norm W_2_norm b_norm b_2_norm Z_2_norm time\n")#write header
dyn_out_file.flush()

output_file_name="MCMC_cnn_%s.txt"%identifier
out_file=open("./results/%s"%output_file_name, "a")
if(verbose):
    print("sampling dataset...\n")

file_mnist=np.load('./datasets/mnist.npz')
if(inputs=='external'):
    X=torch.tensor(file_mnist['x_train'],dtype=torch.float)[:n,None,:,:]
    X_test=torch.tensor(file_mnist['x_test'],dtype=torch.float)[:n_test,None,:,:]
    mean_X=torch.mean(X)
    std_X=torch.std(X)
    X=(X-mean_X)/std_X
    X_test=(X_test-mean_X)/std_X


#generate teacher weights
#seed_teacher=2 (gives satistactory uniformity in label distribution)

torch.manual_seed(seed_teacher)
W_0=torch.randn(size=[C_Z,C_X,H_W,W_W])/torch.sqrt(lambda_W)
b_0=torch.randn(size=[C_Z])/torch.sqrt(lambda_b)
W_2_0=torch.randn(size=[C,d_W_2])/torch.sqrt(lambda_W_2)
b_2_0=torch.randn(size=[C])/torch.sqrt(lambda_b_2)

#sample noise in dataset
torch.manual_seed(seed_noise_dataset)
noise_Z=torch.randn(size=[C_Z,H_Z,W_Z])*torch.sqrt(Delta)
noise_Z_2=torch.randn(size=[n,C_Z,H_Z_2,W_Z_2])*torch.sqrt(Delta)
noise_X_2=torch.randn(size=[n,d_W_2])*torch.sqrt(Delta)
noise_Z_3=torch.randn(size=[n,C])*torch.sqrt(Delta)
if(inputs=='gaussian'):
    X=torch.randn(size=[n,C_X,H_X,W_X])

#sample datasets
Z_0=mc.conv2d_layer(X,W_0,s_y,s_x)+b_0[None,:,None,None]+noise_Z
Z_2_0=mc.average_pool2d(Z_0,H_P,W_P)+noise_Z_2
X_2_0=mc.ReLU(Z_2_0).reshape([Z_2_0.shape[0],-1])+noise_X_2
Z_3_0=X_2_0@(W_2_0.T)+b_2_0[None,:]+noise_Z_3

y_train=CNN_1hl(X,W_0,b_0,s_y,s_x,H_P,W_P,W_2_0,b_2_0,noise_Z,noise_Z_2,noise_X_2,noise_Z_3)
y_noiseless=CNN_1hl_noiseless(X,W_0,b_0,s_y,s_x,H_P,W_P,W_2_0,b_2_0)


#generate test labels
if(inputs=='gaussian'):
    X_test=torch.randn(size=[n_test,C_X,H_X,W_X])
y_test=CNN_1hl_noiseless(X_test,W_0,b_0,s_y,s_x,H_P,W_P,W_2_0,b_2_0)


if(labels=='external'):
    y_train=torch.tensor(file_mnist['y_train'], dtype=torch.long).flatten()[:n]
    y_noiseless=y_train
    y_test=torch.tensor(file_mnist['y_test'],dtype=torch.long).flatten()[:n_test]


#np.savez("./results/ts_setting/red_train_set_mnist_cnn_D4e-3_n60k_seed_teach19_seednoise10000.npz", y=y_train.cpu().numpy(),y_noiseless=y_noiseless.cpu().numpy())
#np.savez("./results/ts_setting/red_test_set_mnist_cnn_D4e-3_n60k_seed_teach19.npz", y=y_test.cpu().numpy())
#np.savez("./results/ts_setting/teacher_weights_mnist_cnn_D4e-3_n60k_seed_teach19.npz",W=W_0.cpu().numpy(), b=b_0.cpu().numpy(),W_2=W_2_0.cpu().numpy(),b_2=b_2_0.cpu().numpy())

if(verbose):
    print("initializing MCMC...")

#pick MCMC initialization
if(initialization=="zero"):
    #zero initialization (seems to work best)

    W=torch.zeros_like(W_0)
    b=torch.zeros_like(b_0)
    Z=torch.zeros_like(Z_0)
    Z_2=torch.zeros_like(Z_2_0)
    X_2=torch.zeros_like(X_2_0)
    W_2=torch.zeros_like(W_2_0)
    b_2=torch.zeros_like(b_2_0)
    Z_3=torch.zeros_like(Z_3_0)

elif(initialization=="informed"):#only available if K=K_0

    W=torch.clone(W_0)
    b=torch.clone(b_0)
    Z=torch.clone(Z_0)
    Z_2=torch.clone(Z_2_0)
    X_2=torch.clone(X_2_0)
    W_2=torch.clone(W_2_0)
    b_2=torch.clone(b_2_0)
    Z_3=torch.clone(Z_3_0)
    if(labels=='external'):
        print("WARNING: informed initialization is meaningless in the case of labels not generated by a teacher")

else:
    raise Exception("Invalid MCMC initialization: "+initialization)



#precomputed quantities
#precomputed quantities
jump_y,jump_x=X.stride()[-2:] #number of positions I have to move in memory to go to the next  3rd, 4th index respectively in X_l.
X_strided_shape=(n,C_X,H_W,W_W,H_Z,W_Z) 
X_strides=X.stride()+(s_y*jump_y,s_x*jump_x)

X_strided=torch.as_strided(X.type(torch.float64),size=X_strided_shape, stride=X_strides)
A_tilde_resc=torch.tensordot(X_strided,X_strided, [[0,4,5],[0,4,5]])

A_resc=A_tilde_resc.reshape([C_X*H_W*W_W,C_X*H_W*W_W]) 
A_resc=A_resc+lambda_W*Delta*torch.eye(C_X*H_W*W_W,dtype=torch.float64)
Cov_W_resc=torch.linalg.inv(A_resc)  
Cholesky_Cov_W_resc=torch.linalg.cholesky(Cov_W_resc).type(torch.float32)  #<---should be passed as argument
Cov_W_resc_XT=torch.tensordot(Cov_W_resc.reshape([C_X*H_W*W_W,C_X,H_W,W_W]),X_strided,[[1,2,3],[1,2,3]]).type(torch.float32) #<--- should be passed as argument


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
W=W.to(device)
b=b.to(device)
Z=Z.to(device)
Z_2=Z_2.to(device)
X_2=X_2.to(device)
W_2=W_2.to(device)
b_2=b_2.to(device)
Z_3=Z_3.to(device)

Delta=Delta.to(device)
X=X.to(device)
X_test=X_test.to(device)
y_train=y_train.to(device)
y_noiseless=y_noiseless.to(device)
y_test=y_test.to(device)
Cholesky_Cov_W_resc=Cholesky_Cov_W_resc.to(device)
Cov_W_resc_XT=Cov_W_resc_XT.to(device)

#variables to track in the dynamics
y_pred_test_histo=torch.zeros(size=[n_test,C],device=device).type(torch.long) #histogram of predicted test labels
flag_thermalized=0
measure_times=list(torch.logspace(0,torch.log10(max_steps),number_measurements)-1.9)+[0 for i in range(20)]
print(measure_times[-1])
torch.manual_seed(seed_MCMC)
start_time=time.time()
if(verbose):
    print("starting the MCMC...\n")
for t in range(max_steps):
    #measure stuff
    if(t>=measure_times[0]):
        measure_times.pop(0)
        y_pred_train=CNN_1hl_noiseless(X,W,b,s_y,s_x,H_P,W_P,W_2,b_2)
        y_pred_test=CNN_1hl_noiseless(X_test,W,b,s_y,s_x,H_P,W_P,W_2,b_2)
        train_acc=torch.sum(y_pred_train==y_train).item()/y_train.shape[0]
        test_acc=torch.sum(y_pred_test==y_test).item()/y_test.shape[0]
        W_norm=torch.linalg.norm(W).item()
        W_2_norm=torch.linalg.norm(W_2).item()
        b_norm=torch.linalg.norm(b).item()
        b_2_norm=torch.linalg.norm(b_2).item()
        Z_2_norm=torch.linalg.norm(Z_2).item()
        elapsed_time=time.time()-start_time
        dyn_out_file.write("%d %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e\n"%(t, test_acc, train_acc, W_norm, W_2_norm, b_norm, b_2_norm, Z_2_norm, elapsed_time))
        dyn_out_file.flush()

    if(t>t_thermalization): 
        flag_thermalized=1
        y_pred_test=CNN_1hl_noiseless(X_test,W,b,s_y,s_x,H_P,W_P,W_2,b_2)
        y_pred_test_histo[torch.arange(n_test),y_pred_test]+=1
    
    #update variables
    Z_3=mc.sample_Z_Lp1_multinomial_probit(X_2@(W_2.T)+b_2[None,:],Z_3,y_train,Delta_Z_Lp1=Delta,precise=True)
    W_2=mc.sample_W_l_fcl(X_2,b_2,Z_3,lambda_W_2,Delta_Z_lp1=Delta)
    b_2=mc.sample_b_l_fcl(W_2,Z_3,X_2,Delta_Z_lp1=Delta,lambda_b_l=lambda_b_2)
    X_2=mc.sample_X_l_fcl(mc.ReLU(Z_2).reshape([n,-1]),W_2,b_2,Z_3,Delta_X_l=Delta,Delta_Z_lp1=Delta)
    Z_2=mc.sample_Z_lp1_relu(fwd_X_l=mc.average_pool2d(Z,H_P,W_P),X_lp1=X_2.reshape([n,C_Z,H_Z_2,W_Z_2]),Delta_Z_lp1=Delta,Delta_X_lp1=Delta,precise=True)    
    Z=mc.sample_X_l_avg_pooling(fwd_Z_l=mc.conv2d_layer(X,W,s_y,s_x)+b[None,:,None,None],X_lp1=Z_2,Delta_X_l=Delta, Delta_X_lp1=Delta)
    W=mc.sample_W_1_conv2d(Cholesky_Cov_W_resc, Cov_W_resc_XT,b,Z,Delta_Z_2=Delta, H_W=H_W, W_W=W_W, C_1=C_X)
    b=mc.sample_b_l_conv2d(X,Z,W,Delta_Z_lp1=Delta,lambda_b_l=lambda_b,stride_y=s_y,stride_x=s_x)

if(verbose):
    print("MCMC finished, preparing output...")
dyn_out_file.close()
BO_estimator_test=torch.argmax(y_pred_test_histo,axis=1)
BO_test_acc=torch.sum(BO_estimator_test==y_test).item()/(y_test.shape[0])
header="t time flag_thermalized BO_test_acc test_acc train_acc input_file seed_teacher seed_noise_dataset seed_student seed_MCMC initialization n_train n_test H_X W_X C_X H_W W_W s_y s_x C_Z H_P W_P C Delta lambda_W lambda_b lambda_W_2 lambda_b_2"
out_file.write(f"{t} {elapsed_time:.1f} {flag_thermalized} {BO_test_acc:1.4e} {test_acc:1.4e} {train_acc:1.4e} {input_split} {seed_teacher} {seed_noise_dataset} {seed_student} {seed_MCMC} {initialization} {n} {n_test} {H_X} {W_X} {C_X} {H_W} {W_W} {s_y} {s_x} {C_Z} {H_P} {W_P} {C} {Delta} {lambda_W} {lambda_b} {lambda_W_2} {lambda_b_2}\n")
out_file.close() #                                                                                                                                         " %d     %.1f         %d                %1.4e     %1.4e     %1.4e      %s             %d             %d                %d            %d          %s         %d    %d    %d  %d  %d     %1.4e      %1.4e      %1.4e         %1.4e       %1.4e    %1.4e    %1.4e        %1.4e       %1.4e        %1.4e        %s                   %s              %d            %d                   %d   "
#np.savez("./results/final_states/config_mnist_mcmc_%s_%s.npz"%(input_split[:-4], identifier), W=W.cpu().numpy(), b=b.cpu().numpy(), Z=Z.cpu().numpy(), X_2=X_2.cpu().numpy(), W_2=W_2.cpu().numpy(), b_2=b_2.cpu().numpy(), Z_2=Z_2.cpu().numpy(), rng_state=torch.get_rng_state().cpu().numpy())
if(verbose):
    print("MCMC executed successfully. Bye!")
