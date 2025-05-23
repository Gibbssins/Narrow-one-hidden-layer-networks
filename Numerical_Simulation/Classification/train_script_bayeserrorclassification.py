# general import
import os
from pathlib import Path
from importlib import reload
import time
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt 
import re

# torch
import torch
torch.set_default_dtype(torch.float64)


# my own
import utils as ut
from utils import check_string, extract_alpha, safe_save, load_files_as_list, compare_dicts, append_unique_line, rank_column_highest_on_diagonal, get_all_parameters, sort_indices, theta_error, hinge_loss 
from network import Committee
import torch.nn.functional as F


import argparse

# read args
parser = argparse.ArgumentParser()

#parser.add_argument("--teacher_student", default=True, action="store_true")
parser.add_argument("--teacher_student", default="True", help="teacher student or memorization")
parser.add_argument("--N", type=int, default=100, help="N")
parser.add_argument("--K", type=int, default=51, help="K")
parser.add_argument("--K_teacher", type=int, default=51, help="K_ teacher")
parser.add_argument("--nonlinearity", type=str, default="erf", help="activation function")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
parser.add_argument("--g", type=float, default=50.0, help="g")
parser.add_argument("--alpha_c", type=float, default=0.0, help="alpha control for the algorithm that looks for the right permutation of the student")
parser.add_argument("--gamma", type=float, default=0., help="gamma")
parser.add_argument("--sigma2_y", type=float, default=1, help="sigma2 y")
parser.add_argument("--T", type=float, default=0., help="T")
parser.add_argument("--num_test", type=int, default=1000, help="num test")
parser.add_argument("--random_v", default=False, action="store_true")
parser.add_argument("--mean_field_v", default=False, action="store_true")
parser.add_argument("--learn_v", default=False, action="store_true")
parser.add_argument("--scale_gamma_T", default=False, action="store_true")
parser.add_argument("--use_bias_W", default=False, action="store_true")
parser.add_argument("--use_bias_v", default=False, action="store_true")
parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
parser.add_argument("--print_every", type=int, default=10, help="test_every")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--momentum", type=float, default=0., help="momentum")
parser.add_argument("--seed", type=int, default=1, help="seed")
parser.add_argument("--seed_all", type=int, default=2, help="seed for all the scripts")
parser.add_argument("--cuda", default=True, action="store_true")
parser.add_argument("--save_dir", default="results_classification", help="save dir")
parser.add_argument("--pertubation",type=float, default=0.0, help="measure of the noise on the teacher")
parser.add_argument("--wise_initialization", default="False", help="initialize close to te teacher")
parser.add_argument("--bias_trick", default="False", help="use the bias trick")
parser.add_argument("--number_best_model", type=int, default=100, help="store last best models")
parser.add_argument("--number_model", type=int, default=10, help="store last 10 models")
parser.add_argument("--perturb_amount", type=float, default=0.0, help="randomness added to the outer layer")
parser.add_argument("--initialize_previousepochs", default="False", help="previous initialization true")
parser.add_argument("--initialize_best_model", default="False", help="previous best initialization true")
parser.add_argument("--store_model_teacher", default="True", help="store the teacher model")






args = parser.parse_args()
# print(vars(args))

# SET MAIN PARAMS
teacher_student = check_string(args.teacher_student)
N = args.N
K = args.K
alpha_c = args.alpha_c
K_teacher = args.K_teacher
nonlinearity = args.nonlinearity
alpha = args.alpha
g=args.g
gamma = args.gamma
sigma2_y = args.sigma2_y
T = args.T
num_test = args.num_test
perturb_amount = args.perturb_amount
random_v = args.random_v
mean_field_v = args.mean_field_v
learn_v = args.learn_v
scale_gamma_T = args.scale_gamma_T
use_bias_W = args.use_bias_W
use_bias_v = args.use_bias_v
num_epochs = args.num_epochs
print_every = args.print_every
lr = args.lr
momentum = args.momentum 
optimizer = 'SGD'
store_model = True
seed = args.seed
seed_all = args.seed_all #*
save_dir = args.save_dir
pertubation = args.pertubation
wise_initialization = check_string(args.wise_initialization) #*
bias_trick = check_string(args.bias_trick) #*
number_best_model = args.number_best_model 
number_model = args.number_model
initialize_best_model = check_string(args.initialize_best_model) #*
store_model_teacher = check_string(args.store_model_teacher)  #*
initialize_previousepochs = check_string(args.initialize_previousepochs) #*
torch.manual_seed(seed_all)

if teacher_student:
    print("We are doing the teacher student simulation")
else:
    print("We are doing the memorization simulation")

###Check the randomness dependence
#print(f"the randomness is {torch.randn(2)}")


# CUDA
cuda = args.cuda
dtype = torch.float64
cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
if cuda:
    torch.cuda.manual_seed_all(seed_all)
    torch.backends.cudnn.benchmark = True
    print("...using", device)
else:
    torch.manual_seed(seed_all)
    print("...using", device)



# SET USEFUL STUFF
# set flag
if teacher_student:
    flag = f"N{N}_K{K}_Kt{K_teacher}_{nonlinearity}_a{alpha}_g{gamma}_scgT{scale_gamma_T}_T{T}_nt{num_test}_rv{random_v}_mfv{mean_field_v}_lv{learn_v}_bW{use_bias_W}_bv{use_bias_v}_lr{lr}_s{seed}"
else:
    flag = f"N{N}_K{K}_{nonlinearity}_a{alpha}_g{gamma}_scgT{scale_gamma_T}_s2y{sigma2_y}_T{T}_rv{random_v}_mfv{mean_field_v}_lv{learn_v}_bW{use_bias_W}_bv{use_bias_v}_lr{lr}_s{seed}"


### Create directory link
try:
    save_dir = f"{save_dir}_teacher_student_{teacher_student}_{nonlinearity}_bias_trick_{bias_trick}_wise_initialization_{wise_initialization}_N{N}_K{K}_T{T}_g_{gamma}_seed_{seed}" 
    save_dir0 = save_dir   
    print(save_dir0)
    dir_results = f"results_classification" + flag
    save_dir += f"/{dir_results}"  
    dump_to = f'{save_dir}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

except:
    save_dir = f"{save_dir}_teacher_student_{teacher_student}_{nonlinearity}_bias_trick_{bias_trick}_wise_initialization_{wise_initialization}_N{N}_K{K}_T{T}_g_{gamma}_seed_{seed}"
    save_dir0 = save_dir
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, save_dir0)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    dir_results = f"results_classification" + flag
    save_dir += f"/{dir_results}"  
    dump_to = f'{save_dir}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

# Usage
file_path = f"{save_dir0}/directories_{lr}_teacher_student_{teacher_student}.txt"
new_line = save_dir
append_unique_line(file_path, new_line)


####Hinge Loss


# convenience quantities
soft = not learn_v
np.random.seed(seed_all)

train_layer = [True, learn_v]
train_bias = [False, False]

gamma_mult = gamma
if T > 0 and scale_gamma_T:
    gamma_mult *= 1 #*T #was commented because the scaling of the regularizer is prop to beta, decommented if otherwise

num_train = int(alpha * N * K) + 1
use_torch_sign = False


# GENERATE DATASET
X = torch.randn(num_train, N).to(device)
print(f"the shape of the data is {X.shape}")
if teacher_student:
    with torch.no_grad():
        net_teacher = Committee(N=N,
                                K=K_teacher,
                                random_v=random_v,
                                mean_field_v=mean_field_v,
                                soft=soft,
                                g=g,
                                nonlinearity=nonlinearity,
                                use_bias_W=use_bias_W,
                                use_bias_v=use_bias_v, 
                                seed=10)
        perturb = torch.randn(K) 
        v_out = net_teacher.v.data + perturb * perturb_amount
        net_teacher.v.data = v_out.to(device)
        ###Check the randomness dependence
        #print(f"the randomness for the second {torch.randn(2)}")
        #print(f"the pertube part of v_out is {perturb[0].item()}")
        #print(f"for the numpy {np.random.normal(K)}")
        #Save the teacher
        teacher_model = get_all_parameters(net_teacher.to("cpu"))
        if store_model_teacher:
            try:
                print("Have you tried this? :-)")
                print(f"the directory {save_dir0+ '/teacher_model.pt'}")
                saved_teacher = torch.load(save_dir0+ '/teacher_model.pt')
                print("You imported it :-)")
                tep = compare_dicts(saved_teacher, teacher_model)
                print(f"the comparison is {tep}")
                if tep:
                    print("the teacher are the same: good :)")
                else:
                    print("the teacher are not the same: good :(")
                    net_teacher.W.data = saved_teacher["W"]
                    net_teacher.v.data = saved_teacher["v"]
                    print(f"we had to change the current teacher but we save it in the later command")
                    safe_save(teacher_model, save_dir0+f'/teacher_model_{flag}.pt')
                    #torch.save(teacher_model, save_dir0+f'/teacher_model_{flag}.pt')
            except:
                print("Primarily saving of the teacher")
                safe_save(teacher_model, save_dir0+'/teacher_model.pt')
                #torch.save(teacher_model, save_dir0+'/teacher_model.pt')
            
        #Device save
        column_norms = net_teacher.W.data.norm(p=2, dim=0, keepdim=True)
        net_teacher.W.data = net_teacher.W.data / column_norms  # No
        net_teacher = net_teacher.to(device)
        _, y = net_teacher(X)
        y_train_sign = torch.sign(y)
        #y = torch.sign(y)
        
        X_test = torch.randn(num_test, N).to(device)
        _, y_test = net_teacher(X_test)
        
        if bias_trick:
            #print (f"the student output prediciton is {output.detach()} after substracting the mean")
            y_test = y_test - y_test.mean()
            y = y - y.mean()
            
        #y_test = torch.sign(y_test)
        y_test_sign = torch.sign(y_test)
        if use_torch_sign:
            y = torch.sign(y)
            y_test = torch.sign(y_test)
else:
    y = torch.randn(num_train) * np.sqrt(sigma2_y)
    y = y.to(device)
    if use_torch_sign:
        y = torch.sign(y)

factsq = math.sqrt(2. * T * lr)

#####Outer Layer
#print(f"the outer layer is {net_teacher.v.data}")     
#print(f"Some weight layer are {net_teacher.v.data[0].item(), net_teacher.v.data[3].item()}")     


# INIT NETWORK
net = Committee(N=N,
                    K=K,
                    random_v=random_v,
                    mean_field_v=mean_field_v,
                    soft=soft,
                    g=g,
                    nonlinearity=nonlinearity,
                    use_bias_W=use_bias_W,
                    use_bias_v=use_bias_v, 
                    seed=seed+10)
dict_net = None

#Normalization of the student weight
column_norms = net.W.data.norm(p=2, dim=0, keepdim=True)
net.W.data = net.W.data / column_norms  # No

#net.to(device)

### Count number of models
print(f"initialize previous epoch :{initialize_previousepochs}")
# init weight properly - frowned upon with T = 0
if not initialize_previousepochs:
    if T > 0 and gamma > 0:
        with torch.no_grad():
            net.W *= math.sqrt(gamma/T) # it was 1/math.sqrt(gamma/T)


if not initialize_previousepochs:
    
    if wise_initialization and teacher_student:
            #print(f"check what is happening {net.W.data==net_teacher.W.data , net.v.data==net_teacher.v.data, net_teacher.v.data }")
            net.W.data = net_teacher.W.data.to(device) + pertubation*net.W.data.to(device) 
    
    #Intit teacher in the same output as the student
    if teacher_student:
        net.v.data = net_teacher.v.data #+ pertubation*net.v.data
    
    # TRAIN
    # init containers and variables
    net.v = net.v.to(device)
    net = net.to(device)
    #print(f"the outer weight of the teacher is {net_teacher.v.data} while for the student is {net.v.data} ")
    
    print("fresh initialization")
    if teacher_student:
        _, output_test = net(X_test)
        
        if bias_trick:
            output_test = output_test - output_test.mean()
        
        if use_torch_sign:
            output_test = torch.sign(output_test)
            
        output_test_sign = torch.sign(output_test)
        err_test = theta_error(output_test_sign, y_test_sign, margin=0.0).sum() / num_test
        print(f"The test error without learning is {err_test}")
    else:
        _, output_train = net(X)
        
        if bias_trick:
            output_train = output_train - output_train.mean()
        
        if use_torch_sign:
            output_train = torch.sign(output_train)
        
        
        err_train = hinge_loss(output_train, y, margin=0.0).sum() / num_train
        print(f"The train error without learning is {err_train}")
    
    results = {}
    losses, errs, normWs, Qst00s, Qst01s, Qss00s, Qss01s = [], [], [], [], [], [], []
    last_best_models_test = [_ for _ in range(number_best_model)]
    last_100_models = [_ for _ in range(number_model)]
    last_best_models_train = [_ for _ in range(number_best_model)]
    best_train_error = 1e10
    best_model = None 
    errs_test = []
    best_err_test = 1e10
    best_model_test = None
    p_test=0
    p_train = 0
    p_model = 0
    
else:
        
    last_best_models_test = [_ for _ in range(number_best_model)]
    last_100_models = [_ for _ in range(number_model)]
    last_best_models_train = [_ for _ in range(number_best_model)]
    path_directories = f"{save_dir0}/directories_{lr}_teacher_student_{teacher_student}.txt"
    directories = load_files_as_list(path_directories)
    p_model = number_model
    path_previousepochs = save_dir
    """for direc in directories:
        alp = extract_alpha(direc)
        if alp == alpha: 
            path_previousepochs = direc
            break"""
        
    if not initialize_best_model:
        try:
            if dict_net==None:
                print(f"the path to previous model is {path_previousepochs}/last_model.pt")
                dict_net = torch.load(f"{path_previousepochs}/last_model.pt")
                print("we initialise from the last model")
            #print("why fail")
            try:
                last_100_model = torch.load(f"{path_previousepochs}/last_{number_model}_models.pt")
                let_model = len(last_100_model)
                last_100_models[:let_model] = last_100_model
                p_model = let_model
            except:
                pass
                
            #print(f"value of p_model {p_model}")
        except:
            if dict_net==None:
                #print(f"the path to previous model is {path_previousepochs}/best_model_based_on_test_error.pt") 
                dict_net = torch.load(f"{path_previousepochs}/best_model_based_on_test_error.pt")
                print("We initialized from the last best model based on test error because the last model was not working")
            last_100_models = [_ for _ in range(number_model)]
            p_model = 0
            #print("An error occur while trying to load all the last model then we set the list to an empty list")
            
    else:
        try:
            if dict_net==None:
                dict_net = torch.load(f"{path_previousepochs}/best_model_based_on_test_error.pt")
                print("We initialize from the last best model based on test error**")
            last_100_model = torch.load(f"{path_previousepochs}/last_{number_model}_models.pt")
            let_model = len(last_100_model)
            last_100_models[:let_model] = last_100_model
            p_model = let_model
        except:
            last_100_models = [_ for _ in range(number_model)]
            p_model = 0
            print("An error occur while trying to load all the last model then we set the list to an empty list")
    try:
        net_W = dict_net["net.W"].to(device)
    except:
        net_W = dict_net["W"].to(device)
    
    try:
        net_v = dict_net["v"] #.to(device)
    except:
            net_v = net_teacher.v.data       
    net.W.data = net_W
    net.v.data = net_v
    net = net.to(device)
    
    if torch.equal(net.v.data.to(device), net_teacher.v.data.to(device)):
        print("the v out are equal")
    else:
        print("the v_out are not equal")
        
    if teacher_student:
        net.v.data = net_teacher.v.data
    net = net.to(device)
    
    print("non fresh initialization")
    if teacher_student:
        _, output_test = net(X_test)
        if bias_trick:
            #print (f"the student output prediciton is {output.detach()} after substracting the mean")
            #y_test = y_test - y_test.mean()
            output_test = output_test -  output_test.mean()
        
        if use_torch_sign:
            output_test = torch.sign(output_test)
        output_test_sign = torch.sign(output_test)
        err_test = theta_error(output_test_sign, y_test_sign).sum().item() / num_test
        print(f"The test error from the last run is {err_test}")
    else:
        _, output_train = net(X)
        if bias_trick:
            #print (f"the student output prediciton is {output.detach()} after substracting the mean")
            #y = y - y.mean()
            output_train = output_train -  output_train.mean()
            
        if use_torch_sign:
            output_train = torch.sing(output_train)
        err_train = hinge_loss(output_train, y, margin=0.0).sum().item() / num_train
        print(f"The train error without learning is {err_train}")
        
    
    print(f"{path_previousepochs}")
    results = torch.load(f"{path_previousepochs}/all_results.pt")
    losses, errs, normWs, Qss00s, Qss01s = results['losses'], results['errs'], results['normWs'], results['DiagOverlapSS'], results['OffOverlapSS']
    if teacher_student: 
        errs_test = results['errs_test']
        try:
            bayes_errs_test = results['bayes_errs_test']
        except:
            bayes_errs_test = []
        best_err_test = min(errs_test)
        Qst00s = results['DiagOverlapST']
        Qst01s = results['OffOverlapST']
    best_train_error = min(errs)
    
    try:
        list_best_trained_model = torch.load(f"{path_previousepochs}/{number_best_model}_best_models_based_on_train_error.pt")
        let_train = len(list_best_trained_model)
        last_best_models_train[:let_train] = list_best_trained_model
        p_train = let_train        
    except:
        p_train = 0
        pass 
    if teacher_student:
        try:
            list_best_tested_model = torch.load(f"{path_previousepochs}/{number_best_model}_best_models_based_on_test_error.pt")
            let_test = len(list_best_tested_model)
            p_test = let_test
            last_best_models_test[:let_test] = list_best_tested_model 
        except:
            p_test = 0
            pass
        
    
    

"""if teacher_student:
    net.v.data += perturb
"""    

        

        
# SET OPTIMIZERS
params = {}
params['soft'] = soft
params['train_layer'] = train_layer
params['train_bias'] = train_bias
params['lr'] = lr
params['momentum'] = momentum 

to_train = net.set_trainable(params)
opt = torch.optim.SGD(to_train)


# measuring time
start_time = time.time()
start_time_total = time.time()

logfile_name = dump_to + "errs.txt"
logfile = open(logfile_name, "w", buffering=1)
to_logfile = "# ep, time, en, err, normW"
if teacher_student:
    to_logfile += ", err test"
ut.log(to_logfile, logfile, print_to_out=False)

#### Check last model first
if initialize_previousepochs:
    #print(f"the previous weight W was {net.W.data}")
    #print(f"the previous weight V was {net.v.data}")
    print(f"the some previous weight V and W are {net.v.data[0].item(), net.W.data[4][0].item()}")
    if teacher_student:
        _, output_test = net(X_test)
        if bias_trick:
            #print (f"the student output prediciton is {output.detach()} after substracting the mean")
            y_test = y_test - y_test.mean()
            output_test = output_test -  output_test.mean()
        
        if use_torch_sign:
            output_test = torch.sign(output_test)   
        output_test_sign = torch.sign(output_test)
        err_test = theta_error(output_test_sign, y_test_sign).sum().item() / num_test
        print(f"The test error from the last run is {err_test}")


# train
batch = 0



"""if bias_trick:
    #print (f"the student output prediciton is {output.detach()} after substracting the mean")
    y = y - y.mean()
    y_test = y_test - y_test.mean()"""
    

for ep in range(1, num_epochs + 1):
    # Forward pass
    hs, output = net(X)
    if bias_trick:
        output = output - output.mean()
    #print(f"at epoch {ep} the output mean of the student network is: {output}")
    if use_torch_sign:
        output = torch.sign(output)
    #print(f"at epoch {ep} the sign of the output for the student network is: {output}")
    
    # Compute loss
    #y = torch.sing(y)
    #err =  thetha_loss(output, y, margin=0.0).sum() 
    err = hinge_loss(output, y, margin=0.0).sum()
    #err =  ((output- y)**2).sum()
    normW = (net.W**2).sum()
    loss = 0.5 * err + 0.5 * gamma_mult * normW
    #print(f"At epoch {ep} the loss is {loss}")

    #Backward pass (no retain_graph=True)
    opt.zero_grad()
    loss.backward()  # Remove retain_graph=True
    opt.step()
    

    # Add noise (reduce T/lr if noise is too large)
    if T > 0:
        with torch.no_grad():
            noise = torch.randn(N, K).to(device) * factsq
            net.W.data += noise  # Direct in-place addition
    
    #print(f"Look at the weight {net.W.data[2:6]}")  
    with torch.no_grad():
        column_norms = net.W.data.norm(p=2, dim=0, keepdim=True)
        net.W.data = net.W.data / column_norms  # No 
    # Record loss (pre-noise)
    #losses.append(loss.item() / X.shape[0])
    
    
    #Check the gradient
    #print(f"Look at the gradient of the parameter {net.W.requires_grad}")
    

    # exit if NaN loss
    if np.isnan(loss.item()):
        print(f"the value of the loss is {loss.item()} err is {err} normW is {normW} student output {output} teacher output{y}")
        raise ValueError("NaN happened")

    # record loss
    losses.append(loss.item()/X.shape[0])
    add_err = err.item()/X.shape[0]
    errs.append(add_err)
    normWs.append(normW.item()/N/K)

    
    if teacher_student:
        
        _, output_test = net(X_test)
        if bias_trick:
           output_test = output_test - output_test.mean()
        if use_torch_sign:
            output_test = torch.sign(output_test)
        output_test_sign = torch.sign(output_test)
        err_test = theta_error(output_test_sign, y_test_sign, margin=0.0).sum().item() / num_test 
        errs_test.append(err_test)
    
    
        
    ## 
    s_W = net.W.data.clone()
    if alpha>=alpha_c:
        s_W = sort_indices(net_teacher.W.data.T.clone(), s_W.T).T
    OverlapSS = torch.matmul(s_W.T, s_W) / N
    DiagOverlapSS = torch.diag(OverlapSS)
    OffOverlapSS = OverlapSS[~torch.eye(*OverlapSS.shape, dtype=bool)]
    Qss01 = OffOverlapSS.sum().item()/(K * (K - 1))
    Qss00 = (DiagOverlapSS).mean().item()
    Qss00s.append(Qss00)
    Qss01s.append(Qss01)
    
    if teacher_student: 
        
        OverlapST = torch.matmul(s_W.T, net_teacher.W.data) / N
         
        if perturb_amount==0.0:
            print("I am using the optimization algorithm")
            OverlapST = rank_column_highest_on_diagonal(OverlapST)
           
        DiagOverlapST = torch.diag(OverlapST)
        
        #print(f"the shape of the diagonal array of the teacher-student overlap is {DiagOverlapST.shape} and the shape of the overlap teacher-student is {OverlapST.shape}")
        #print(f"check eye matrix {torch.eye(*OverlapST.shape, dtype=bool)}  and the shape is {OverlapST.shape}")
        
        OffOverlapST = OverlapST[~torch.eye(*OverlapST.shape, dtype=bool)]
        Qst01 = OffOverlapST.sum().item()/(K * (K - 1))
        Qst00 = (DiagOverlapST).mean().item()
        Qst00s.append(Qst00)
        Qst01s.append(Qst01)
        #img = plt.imshow(OverlapST.to('cpu').numpy())
        #print(OverlapST)
    
    
    #Save latest model
    if p_model==number_model: 
        p_model = 0
        if store_model: 
           safe_save(last_100_models, dump_to + f'last_{number_model}_models.pt')
           #torch.save(last_100_models, dump_to + f'last_{number_model}_models.pt')
    #print(p_model, number_model)
    last_100_models[p_model] = net.W.data
    p_model +=1
    last_model = get_all_parameters(net)
    if store_model:
        #print(f"We saved the last model")
        safe_save(last_model, dump_to + 'last_model.pt')
        #torch.save(last_model, dump_to + 'last_model.pt')
        
    
    # print
    if ep % print_every == 0:
        # measure time & print
        elapsed_time = time.time() - start_time
        start_time = time.time()
        toprint = 'teacher_student: {} p_train: {} ep: {} time: {:.2f} en: {:.8f} err: {:.8f} normW: {:.8f}'.format(teacher_student,
                                                                                            p_train,
                                                                                                ep,
                                                                                    elapsed_time/60,
                                                                                    loss.item()*2/N,
                                                                                    err.item()/N,
                                                                                    normW.item()/N/K)
        
        # test
        if teacher_student:
            toprint += 'err test: {:.8f}'.format(err_test)
            # check if improved test
            improved_test = False
            improved_test = err_test < best_err_test
            if improved_test:
                if p_test==number_best_model: 
                    p_test = 0
                    if store_model: 
                        safe_save(last_best_models_test, dump_to + f'{number_best_model}_best_models_based_on_test_error.pt')
                        #torch.save(last_best_models_test, dump_to + f'{number_best_model}_best_models_based_on_test_error.pt')
                last_best_models_test[p_test] = net.W.data.clone().detach()
                best_err_test = err_test
                best_model_test = get_all_parameters(net)
                if store_model:
                    safe_save(best_model_test, dump_to + 'best_model_based_on_test_error.pt')
                    #torch.save(best_model_test, dump_to + 'best_model_based_on_test_error.pt')
                p_test +=1

        # check if improved train
        try:
            if p_train==number_best_model:
                #print("I am saving the bayes error")
                new_net = net
                for pred in range(len(last_best_models_train)):
                    new_net.W.data = last_best_models_train[pred]
                    _, output_testg = new_net(X_test)
                    if bias_trick:
                        output_testg = output_testg - output_testg.mean()
                    try:
                        output_testgf += output_testg
                    except:
                        output_testgf = output_testg
                bayesian_prediction = output_testgf/len(last_best_models_train)
                if use_torch_sign:
                    bayesian_prediction = torch.sign(bayesian_prediction)
                bayesian_prediction_sign = torch.sign(bayesian_prediction)
                bayes_err_test = theta_error(bayesian_prediction_sign, y_test_sign).sum().item() / num_test 
                bayes_errs_test.append(bayes_err_test)
                del output_testgf
                p_train=0
                print(f"the bayesian prediction is {bayesian_prediction[:3].detach()} and the true predicition is {y_test[:3].detach()} and the last prediction was {output_testg[:3].detach()}")
            else:
                #check if improved train
                if p_train==number_best_model-1: 
                        if store_model: 
                            safe_save(last_best_models_train, dump_to + f'{number_best_model}_best_models_based_on_train_error.pt')
                            #torch.save(last_best_models_train, dump_to + f'{number_best_model}_best_models_based_on_train_error.pt')
                last_best_models_train[p_train] = net.W.data.clone().detach()
                #print(f"p_train {p_train} last_best_models_train[p_train] {last_best_models_train[p_train][0,:3]} last_best_models_train[3] {last_best_models_train[3][0,:3]}")
                #print(f"p_train {p_train} net.W.data {net.W.data[0,:3]} last_best_models_train[p_train] {last_best_models_train[p_train][0,:3]} last_best_models_train[p_train-1] {last_best_models_train[p_train-1][0,:3]} initialize the list and improved_train {improved_train} best_train_error {best_train_error} train_error {add_err}")

                improve_train_error = best_train_error <= add_err
                if improve_train_error:
                    best_model = get_all_parameters(net) 
                    if store_model:
                        safe_save(best_model, dump_to +'best_model_based_on_train_error.pt') 
                        #torch.save(best_model, dump_to + 'best_model_based_on_train_error.pt') 
                    best_train_error = add_err
                p_train +=1   
                
        except:
            #print("could not run the bayes error but I am saving the data model")
            #check if improved train
            if p_train==number_best_model: 
                    p_train=0
                    if store_model: 
                        safe_save(last_best_models_train, dump_to + f'{number_best_model}_best_models_based_on_train_error.pt')
                        #torch.save(last_best_models_train, dump_to + f'{number_best_model}_best_models_based_on_train_error.pt')
                        
            last_best_models_train[p_train] = net.W.data.clone().detach()
            improve_train_error = best_train_error <= add_err
            if improve_train_error:
                best_model = get_all_parameters(net)
                if store_model:
                    safe_save(best_model, dump_to + 'best_model_based_on_train_error.pt')
                    #torch.save(best_model, dump_to + 'best_model_based_on_train_error.pt')
                best_train_error = add_err
            p_train +=1
            pass    
                
            

        # print if improved
        improve_train_error = best_train_error <= add_err
        if improve_train_error:
            toprint += u' \u2193'
        if teacher_student and improved_test:
                toprint += ' *'

        print(toprint)
        
        # log to file
        to_logfile = '{} {:.2f} {:.8f} {:.8f} {:.8f}'.format(ep,
                                                             elapsed_time/60,
                                                             loss.item()*2/N,
                                                             err.item()/N,
                                                             normW.item()/N/K)
        if teacher_student:
            to_logfile += ' {:.8f}'.format(err_test)

        ut.log(to_logfile, logfile, print_to_out=False)

    result = {'losses': losses, 'errs': errs, 'normWs':normWs}
    result['DiagOverlapSS'] = Qss00s 
    result['OffOverlapSS'] = Qss01s
    if teacher_student: 
        result['errs_test'] = errs_test
        
        try:
            result['bayes_errs_test'] = bayes_errs_test
        except:
            pass
        result['DiagOverlapST'] = Qst00s 
        result['OffOverlapST'] = Qst01s
        
    safe_save(result, dump_to + 'all_results.pt')
    #torch.save(result, dump_to + 'all_results.pt')


elapsed_time = time.time() - start_time_total
print("done for all")
print("elapsed:", str(datetime.timedelta(seconds=elapsed_time)))