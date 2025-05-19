import os 
import torch
import argparse
from ps_utils_data import save_data_paper, format_string, load_files_as_list, extract_alpha, extract_directory, upload_dat, extract_hyperarameters, upload_last_model, upload_best_model_based_on_test, generate_colors, make_of_dict, plot_exp_theo, dynamic_training, upload_theoretical_data
from ps_utils_data import generate_colors, make_of_dict, plot_exp_theo, dynamic_training, upload_theoretical_data, safe_save_txt



# read args
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning_rate")
args = parser.parse_args()
lr = args.lr


wise_initialization = True
path1 = f"directories_{lr}_teacher_student_True.txt"
list_directories1 = load_files_as_list(path1)
list_directories = list_directories1  
model_teacher = torch.load("teacher_model.pt")
list_directories2 = [extract_directory(dir) for dir in list_directories]
alpha_list = [extract_alpha(direct) for direct in list_directories ]
print(f"the values of alpha are {alpha_list}")
if len(list_directories)!=0:
    activ_func, gamma, T, N, K, wise_initialization = extract_hyperarameters(list_directories[0])
print(activ_func, gamma, T, N, K)
all_results_alpha = upload_dat(list_directories2)
last_models = upload_last_model(list_directories2)
last_best_model_based_on_test =upload_best_model_based_on_test(list_directories2)
# Example usage
num_colors = len(alpha_list)
color_list = generate_colors(num_colors)
print(color_list)

#Import experimental data last p of them for which we will compute mean and std
dict_list, plot_list_alpha, subkeyset = make_of_dict(all_results_alpha,p=-500)
#print(f"the element of dictionnary are {dict_list.keys()}")
theo_directory=["Generali_Error_theoric_specialized_1.txt", "Overlap_quantities_specialized_1.txt"]
plot_exp_theo(subkeyset, dict_list, theo_directory,N,K,T,gamma,activ_func,lr,wise_initialization)
plot_exp_theo(subkeyset, dict_list, theo_directory,N,K,T,gamma,activ_func,lr,wise_initialization,semilog=False)

####Dynamics of learning
#dynamic_training(all_results_alpha, subkeyset, plot_list_alpha, N, K, T, gamma, activ_func, lr,wise_initialization)

## Keys quantity
subkeyset1 = {"errs_test", "normWs", "DiagOverlapST", "OffOverlapST", "OffOverlapSS", "bayes_errs_test"}
plot_exp_theo(subkeyset1, dict_list, theo_directory,N,K,T,gamma,activ_func,lr,wise_initialization,semilog=False)

dic_theo = upload_theoretical_data([theo_directory], K)
names = f'N{format_string(f"{N}")}_K{format_string(f"{K}")}_T{format_string(f"{T}")}_gamma{format_string(f"{gamma}")}_activ_funct{activ_func}_wise_initialization{wise_initialization}'
name = ["experiment", "specialized 1"]
subkeyset2 = {'errs_test', 'DiagOverlapST','OffOverlapSS', 'OffOverlapST', 'errs', 'losses', 'normWs', "bayes_errs_test"}
for keys in subkeyset2:
    dict = {}
    p=0
    #print(f"the keys for which there is an error is {keys}")

    for i in range(len(name)):
        array = dict_list[keys]
        #print(f"i and keys {i,keys}")
        if i==0:
            dict[name[i]] = array
        else:
            try:
                dict[name[i]] = [dic_theo["alpha"][p], dic_theo[keys][p]]
                p+=1
            except:
                pass 
    #print(f"i and keys dict {i,keys, dict}")        
    save_data_paper(keys, name, dict, lr, names)
    
    
    

list_quantity_plot = ["errs_test", "normWs", "DiagOverlapST", "OffOverlapST", "OffOverlapSS", "bayes_errs_test"]
path = f"experiementadatafor_lr{lr}"+names
# Create the directory if it doesn't exist
os.makedirs(path, exist_ok=True)


for key in list_quantity_plot:
    check = torch.load(f"data_for_paper{names}/{key}_{lr}.pt")
    quantity = check[key]['experiment']
    path_directory_quantity = os.path.join(path, f"{key}_{lr}.txt")
    safe_save_txt(quantity, path_directory_quantity)
    print(f"{key} was save 1")
    #np.savetxt(quantity, path_directory_quantity)