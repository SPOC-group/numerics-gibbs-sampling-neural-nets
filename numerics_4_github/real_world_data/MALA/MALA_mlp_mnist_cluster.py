# coding: utf-8

#this program implements an MCMC for a two layer network and trains it in the teacher student setting
#the dynamical output gets saved in ./results/dyn/
#the final output gets saved in ./results/MCMC_2_layer_NN_regr_teach_stud.txt
#the program runs mainly on the GPU 
# !!! At the moment there is only support for ReLU activation !!!
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import argparse
import time
import os 

class Net(nn.Module):
    def __init__(self, layer_sizes, informed_weights_dict=None,std_weight_dict=None):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1], bias=True)
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2], bias=True)

        if informed_weights_dict==None:
            for name,W in self.named_parameters():
                W.data=torch.randn(size=W.data.shape, requires_grad=True)*std_weight_dict[name]
        else:
            print("WOW! a well knowlegeable student")
            self.load_state_dict(informed_weights_dict)
            
    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x

class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, num_classes,informed_weights_dict=None,std_weight_dict=None):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=4, stride=2,padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(72,num_classes,bias=True) #this dimension only works for mnist as input

        if informed_weights_dict==None:
            for name,W in self.named_parameters():
                W.data=torch.randn(size=W.data.shape, requires_grad=True)*std_weight_dict[name]
        else:
            print("WOW! a well knowlegeable student")
            self.load_state_dict(informed_weights_dict)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.relu(out)
        out = self.fc(out)
        return out

def log_posterior(y_pred,y,model,lambda_dict,Delta,loss): #in this case y_pred sono le preattivazioni dell'ultimo layer.
    logP=torch.tensor(0.,requires_grad=True)
    for name,W in model.named_parameters():
        logP= logP - torch.sum(W**2)*lambda_dict[name]
    return 0.5*(logP-loss(y_pred,y)/Delta) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type(torch.FloatTensor)
#parse
verbose=True
if(verbose):
    print("current device: ", device, "\n")
parser = argparse.ArgumentParser()
parser.add_argument("-Delta", "--Delta",help="posterior is P(W)exp(-L/2*Delta), where L is the training loss")
parser.add_argument("-lr", "--learning_rate",help="learning rate")
parser.add_argument("-id", "--identifier", default="",help="string to identify the run or group or runs")
parser.add_argument("-tmax", "--max_steps",help="max number of MALA steps")
parser.add_argument("-nmeas", "--number_of_measurements",help="number of measurements during the simulation")

args = parser.parse_args()
identifier=args.identifier
Delta=torch.tensor(float(args.Delta),requires_grad=False)
learning_rate=torch.tensor(float(args.learning_rate))
tmax=torch.tensor(int(args.max_steps))
number_measurements=torch.tensor(int(args.number_of_measurements))
#simulation outputs
#For each run of the algorithm one dynamics file is created. The dynamics file is identified by the name of the script that was used to launch the simulation and the identifier.
#In addition to the dynamics file another global output file is opened in append mode. This global file only contains information regarding the whole simulation (e.g. average quantities)
#Simulations with the same identifier will be written in the same global file. Hence one should use the identifier to mark simulations that one wants to compare, or that belong to the same set of experiments.

if(verbose):
    print("initializing output files...")
os.makedirs("./results", exist_ok=True)
os.makedirs("./results/dyn",exist_ok=True)
dyn_output_file_name="dyn_MALA_mlp_mnist_D%.1e_lr%.1e"%(Delta.item(),learning_rate.item())+"_"+identifier+".txt"
dyn_out_file=open("./results/dyn/%s"%dyn_output_file_name, "w")
dyn_out_file.write("t test_loss train_loss W_norm W_2_norm b_norm b_2_norm time\n")#write header

output_file_name="MALA_mlp_mnist_%s.txt"%identifier
out_file=open("./results/%s"%output_file_name, "a")

if(verbose):
    print("loading mnist dataset...")

#load mnist dataset
n=60000
n_test=5000

file_mnist=np.load("./datasets/mnist.npz",allow_pickle=True)
X_train=torch.tensor(file_mnist["x_train"]) #use only half of the training set
X_test=torch.tensor(file_mnist['x_test']) #use only half of the test set
X_train=X_train.reshape([60000,28*28])[:n]/255
X_test=X_test.reshape([10000,28*28])[:n_test]/255
mean_X=torch.mean(X_train)
std_X=torch.std(X_train)

X_train=(X_train-mean_X)/std_X
X_test=(X_test-mean_X)/std_X
y_train=torch.tensor(file_mnist['y_train']).type(torch.long).flatten()[:n]
y_test=torch.tensor(file_mnist['y_test']).type(torch.long).flatten()[:n_test]

X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
####
if(verbose):
    print("initializing network...")
seed=123
torch.manual_seed(seed)
#initialize student
loss = nn.CrossEntropyLoss(reduction='sum')
#define network
layer_sizes = [784,12,10]
lambda_dict={'l1.weight': torch.tensor(layer_sizes[0]),'l2.weight': torch.tensor(layer_sizes[1]),'l1.bias': torch.tensor(layer_sizes[0]),'l2.bias':torch.tensor(layer_sizes[1])} #dictionary with the priors
std=1e-4
std_dict={'l1.weight':std,'l1.bias':std,'l2.weight': std, 'l2.bias':std}
student = Net(layer_sizes,None,std_dict)
criterion=lambda y_pred,y:1-torch.sum(y==y_pred)/len(y) #only used for performances computation
if(verbose):
    print("moving variables to %s..."%str(device))
#moving things to the GPU if possible
#move things to device
student=student.to(device)
X_train=X_train.to(device)
y_train=y_train.to(device)
X_test=X_test.to(device)
y_test=y_test.to(device)
learning_rate=learning_rate.to(device)
Delta=Delta.to(device)
for key in lambda_dict:
    lambda_dict[key]=lambda_dict[key].to(device)

#initialize variables
optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate)
grad_x={}
grad_x_sq_norm=torch.tensor(0.0,requires_grad=False).to(device)
optimizer.zero_grad()
train_pred=student(X_train)
logP_x=log_posterior(train_pred,y_train,student,lambda_dict,Delta,loss)
logP_x.backward()

with torch.no_grad():   
    for name, W in student.named_parameters():
        grad_x[name]=W.grad.detach().clone()
        grad_x_sq_norm+=torch.sum(W.grad**2)


delta_W={}
logP_xp=torch.tensor(0,requires_grad=False).to(device)
grad_xp={}
##############################

#variables to track
train_loss=[]
test_loss=[]
W_norm=[]
W_2_norm=[]
b_norm=[]
b_2_norm=[]
accepted=[]

measure_times=list(torch.logspace(0,torch.log10(tmax),number_measurements)-1)+[1e100]
start_time=time.time()
if(verbose):
    print("starting the MCMC...\n")
for t in range(tmax):
    with torch.no_grad():
        #measure stuff
        if(t>=measure_times[0]):
            measure_times.pop(0)
            elapsed_time=time.time()-start_time
            test_pred=torch.argmax(student(X_test),axis=1)
            train_pred=torch.argmax(student(X_train),axis=1) 
            train_loss=criterion(train_pred,y_train).item()
            test_loss=criterion(test_pred,y_test).item()
            W_norm,b_norm,W_2_norm,b_2_norm=[torch.linalg.norm(param).cpu().numpy().item() for param in student.parameters()]
            dyn_out_file.write("%d %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e %1.4e\n"%(t,test_loss,train_loss,W_norm,W_2_norm,b_norm,b_2_norm,elapsed_time))
        # end measure_stuff #

        #do the langevin step using the previous gradient
        for name, W in student.named_parameters():
            delta_W[name]=learning_rate*grad_x[name]+torch.sqrt(2*learning_rate)*torch.randn_like(W)
            W+=delta_W[name] #propose a step 

    #compute gradient of logP in the new weights       
    optimizer.zero_grad()
    outputs = student(X_train) 
    logP_xp = log_posterior(outputs, y_train,student,lambda_dict, Delta,loss)
    logP_xp.backward()
    with torch.no_grad():
        delta_W_dot_grad=0
        grad_xp_sq_norm=0
        for name, W in student.named_parameters():
            delta_W_dot_grad+=torch.sum((grad_x[name]+W.grad)*delta_W[name])
            grad_xp[name]=W.grad
            grad_xp_sq_norm+=torch.sum(W.grad**2)
        
        log_P_accept=logP_xp-logP_x-learning_rate*(grad_xp_sq_norm-grad_x_sq_norm)/4-0.5*delta_W_dot_grad #computing the acceptance probability
        if(torch.log(torch.rand(1,device=device))<log_P_accept): #accept
            grad_x=copy.deepcopy(grad_xp)
            grad_x_sq_norm=grad_xp_sq_norm.detach().clone()
            logP_x=logP_xp.detach().clone()
            accepted.append(True)
        else:
            accepted.append(False)
            for name, W in student.named_parameters(): #reject
                W-=delta_W[name]

if(verbose):
    print("MCMC finished, preparing output...")
dyn_out_file.close()
elapsed_time=time.time()-start_time
out_file.write(f"{tmax} {t} {elapsed_time:1.4e} {seed} {test_loss:1.4e} {train_loss:1.4e} {std:1.4e} {Delta.cpu().numpy().item():1.4e} {identifier} {learning_rate.cpu().numpy().item():1.4e} {number_measurements}\n")

#header="max_steps num_steps final_time test_loss_last_W train_loss_last_W  std_init_weights Delta identifier learning_rate number_measurements\n"


out_file.close()
if(verbose):
    print("MALA executed successfully. Bye!")
