import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import torch as tch 
import re
import os 
import torch




def process_complex_file(file_path):
    # Function to parse and remove the imaginary part
    def process_number(s):
        try:
            # Replace 'I' with 'j' to match the standard Python complex number format
            s = s.replace('I', 'j')
            # Convert to a complex number
            num = complex(s)
            # Return only the real part
            return num.real
        except ValueError:
            # If parsing fails, return None (or handle as needed)
            return None

    # List to store lists of real numbers for each line
    all_real_numbers = []
    previous_length = None

    # Read and process the file
    with open(file_path, 'r') as f:
        for line in f:
            line_real_numbers = []  # Create a new list for each line
            elements = line.split(',')  # Split each line by commas
            for item in elements:
                real_part = process_number(item.strip())  # Remove any surrounding whitespace
                if real_part is not None:  # Only keep valid numbers
                    line_real_numbers.append(real_part)

            # Only add to all_real_numbers if the line length is consistent with previous lines
            if previous_length is None or len(line_real_numbers) == previous_length:
                all_real_numbers.append(line_real_numbers)
                previous_length = len(line_real_numbers)  # Update length to current line's length
            else:
                #print(f"Skipping line with inconsistent number of elements: {line.strip()}")
                pass

    # Convert to numpy array if all lines have the same length
    if all_real_numbers and len(all_real_numbers[0]) == previous_length:
        return np.array(all_real_numbers)
    else:
        return None

    
def format_string(input_str):
    try:
        return input_str.replace(".", "p")
    except:
        input_str = f"{input_str}"
        return input_str.replace(".", "p") 

def load_files_as_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def extract_alpha(string):
    match = re.search(r'_a(\d+\.\d+)_', string)
    if match:
        return float(match.group(1))
    else:
        return None
def extract_directory(string):
    return string.split('/')[-1] 

def extract_hyperarameters(string):
        
    match = re.search(r'True_([^_]+)_', string)
    if match:
        activ_func = str(match.group(1))
    else:
        match = re.search(r'False_([^_]+)_', string)
        if match:
            activ_func = str(match.group(1))
        else:
            activ_func = None 
            
    match = re.search(r'wise_initialization_([^_]+)_', string)
    if match:
        init = str(match.group(1))
    else:
        init = None 
        
    match = re.search(r'_g(\d+\.\d+)_', string)
    if match:
        gamma = float(match.group(1))
    else:
        gamma = None 
        
    match = re.search(r'_T(\d+\.\d+)_', string)
    if match:
        T = float(match.group(1))
    else:
        T = None
        
    match = re.search(r'_K(\d+)_', string)
    if match:
        K = int(match.group(1))
    else:
        K = None 
        
    match = re.search(r'_N(\d+)_', string)
    if match:
        N = int(match.group(1))
    else:
        N = None
    
    return activ_func, gamma, T, N, K, init 


def upload_dat(directories):
    all_results_alpha = { }
    for directory in directories:
        #try:
        all_results_alpha[f"result_{extract_alpha(directory)}"] = tch.load(f"{directory}/all_results.pt") 
        #except:
        #    pass
    return all_results_alpha 

def upload_last_model(directories):
    last_models = { }
    for directory in directories:
        try:
            last_models[f"result_{extract_alpha(directory)}"] = tch.load(f"{directory}/last_10_models.pt") 
        except:
            pass
    return last_models

def upload_best_model_based_on_test(directories):
    last_best_model_based_on_test = { }
    for directory in directories:
        try:
            last_best_model_based_on_test[f"result_{extract_alpha(directory)}"] = tch.load(f"{directory}/10_best_models_based_on_test_error.pt") 
        except:
            pass
    return last_best_model_based_on_test 

def generate_colors(n):
    """
    Generate a list of 'n' distinct colors that are suitable for matplotlib and very readable.
    
    Parameters:
    n (int): The number of colors to generate.
    
    Returns:
    List of color hex codes.
    """
    if n <= 20:
        # Use 'tab20' colormap for up to 20 colors
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(n)]
    else:
        # Use 'hsv' colormap to generate more than 20 colors
        colors = [plt.cm.hsv(i / n) for i in range(n)]
    
    # Convert RGBA colors to hex
    colors = [ matplotlib.colors.rgb2hex(color[:3]) for color in colors]
    return colors

def extract_alpha2(string):
    match = re.search(r'lt_(\d+\.\d+)', string)
    if match:
        return float(match.group(1))
    else:
        return None
    
    
def make_of_dict(all_results_alpha,p = -1000):
    #print(f"the subkey is {all_results_alpha}")
    
    if not all_results_alpha:
        print("The dictionary is empty.")
    else:
        print("The dictionary is not empty.")
    
    dict_list = {}
    i=0
    subkeyset = {}
    plot_list_alpha = []

    for key in list(all_results_alpha.keys()):
        
        #print(f"the subkey is {subkey}")
        alpha = extract_alpha2(key)
        #print(alpha)
        plot_list_alpha.append(alpha)
        for subkey in list(all_results_alpha[key].keys()):
            
            #if subkey=='last_temperature':
            #    continue
            #print(f"the subkey is {subkey}")
            try:
                subkeyset.add(subkey)
            except:
                subkeyset = {subkey}
                
            try:
                y = all_results_alpha[key][subkey][p:]
                mean, std = np.mean(y).item(), np.std(y).item()
                dict_list[subkey].append([alpha, mean, std]) 
                
            except:
                dict_list[subkey] = []
                y = all_results_alpha[key][subkey][p:]
                mean, std = np.mean(y).item(), np.std(y).item()
                dict_list[subkey].append([alpha, mean, std]) 
        i+=1 

    for subkey in subkeyset:
        array = np.array(dict_list[subkey])
        dict_list[subkey]  = array[array[:,0].argsort()]
    return dict_list, plot_list_alpha, subkeyset

    
def plot_exp_theo(subkeyset, dict_list, theo_directory,N,K,T,gamma,activ_func, lr, wise_initialization, semilog=True, loglog=False,figsize=(20, 4), name="all_quantity"):
    list_quantity_plot = list(subkeyset)
    fig, ax = plt.subplots(1, len(list_quantity_plot), figsize=figsize)
    
    try:
        error_quad_2 = process_complex_file(theo_directory[0])
        error_quad_2_sorted = error_quad_2[error_quad_2[:,0].argsort()]
        order_param_erf_2 = process_complex_file(theo_directory[-1])
        order_param_erf_2_sorted = order_param_erf_2[order_param_erf_2[:,0].argsort()]
        alpha, q0, q1, m0, m1, v0, v1 = order_param_erf_2_sorted[:,0], order_param_erf_2_sorted[:,1], order_param_erf_2_sorted[:,2],order_param_erf_2_sorted[:,3],order_param_erf_2_sorted[:,4],order_param_erf_2_sorted[:,5],order_param_erf_2_sorted[:,6] 

        theo_normWs = q0 + v0 + (q1 + v1)/K

        theo_stm1 = (m1)/K
        theo_ssq1 = (q1 + v1)/K
        theo_diag_ts = m0 + m1/K
    except:
        print(f"none theoretical data to display")

    l = 0
    for keys in list_quantity_plot:
        
        array = dict_list[keys]
        start=0    
        x_el = array[:,0][start:]
        #print(x_el)
        y_el = array[:,1][start:]
        y_err = array[:,2][start:]
        ax[l].errorbar( x_el, y_el, yerr= y_err , fmt ='.--', capsize=5, alpha=1, label="exp")
        ax[l].set_title(f"{keys}")
        #ax[l].set_xlim(xmin=1.2)
        
        try:
            if keys == "errs_test":
                ax[l].plot(error_quad_2_sorted[:,0][start:], 2*error_quad_2_sorted[:,1][start:],label="theo",color="red")
                ax[l].legend()
                if loglog: ax[l].set_xscale("log")
                if semilog: ax[l].set_yscale("log")
                
            if keys == "normWs":
                ax[l].plot(alpha, theo_normWs,label="theo",color="red")
                ax[l].legend()
                if loglog: ax[l].set_xscale("log")
                if semilog: ax[l].set_yscale("log")
                
                
            if keys == "DiagOverlapST":
                ax[l].plot(alpha, theo_diag_ts,label="theo",color="red")
                ax[l].legend()
                if loglog: ax[l].set_xscale("log")
                if semilog: ax[l].set_yscale("log")
                
            if keys == "DiagOverlapSS":
                ax[l].plot(alpha, theo_normWs,label="theo",color="red")
                ax[l].legend()
                if loglog: ax[l].set_xscale("log")
                if semilog: ax[l].set_yscale("log")
                
            if keys == "OffOverlapST":
                ax[l].plot(alpha, theo_stm1,label="theo",color="red")
                ax[l].legend()
                if loglog: ax[l].set_xscale("log")
                if semilog: ax[l].set_yscale("log")
                
            if keys == "OffOverlapSS":
                ax[l].plot(alpha, theo_ssq1,label="theo",color="red")
                ax[l].legend()
                if loglog: ax[l].set_xscale("log")
                if semilog: ax[l].set_yscale("log")  
        except:
            pass
        l+=1
    elements="p"
    for el in list_quantity_plot:
        elements=elements+'_'+el
    names=f'N{N}_K{K}_T{format_string(T)}_gamma{format_string(gamma)}_activ_funct_{activ_func}_lr{format_string(lr)}wiseInit_{wise_initialization}_semilog{semilog}'    
    fig.suptitle(f'N={N}, K={K}, T={T}, γ={gamma}, activ_funct ={activ_func} lr={lr} wiseInit={wise_initialization}')
    path_save="figures"
    try:
        plt.savefig(f"{path_save}/{name}_{names}.png")
    except:
        os.makedirs(path_save, exist_ok=True)
        plt.savefig(f"{path_save}/{name}_{names}.png")
    
def dynamic_training(all_results_alpha, subkeyset, plot_list_alpha, N, K, T, gamma, activ_func, lr, wise_initialization, figsize=(20, 14),name="dynamic"):
    fig, ax = plt.subplots(len(plot_list_alpha), len(subkeyset), figsize=figsize)

    i,j = 0, 0
    for alpha in plot_list_alpha:
        
        key = f"result_{alpha}"
        
        for subkey in list(all_results_alpha[key].keys()):
            
            #if subkey=='last_temperature':
            #    continue
            
            try:
                p=0
                y = all_results_alpha[key][subkey][p:]
                ax[i,j].plot(np.arange(len(y)), y)
            except:
                ax[i,j].plot(np.arange(len(all_results_alpha[key][subkey])), all_results_alpha[key][subkey])
                
            title = f"{subkey}"
            if j==0:
                title += f" and alpha = {alpha}"
            ax[i,j].set_title(f"{title}")
            ax[i,j].set_xscale("log")
            #ax[i,j].set_yscale("log")
            
            j+=1
        j=0   
        i+=1
    names=f'N{N}_K{K}_T{format_string(T)}_gamma{format_string(gamma)}_activ_funct_{activ_func}_lr{format_string(lr)}wiseInit_{wise_initialization}'    
    fig.suptitle(f'N={N}, K={K}, T={T}, γ={gamma}, activ_funct ={activ_func} lr={lr} wiseInit={wise_initialization}')
    path_save="figures"
    try:
        plt.savefig(f"{path_save}/{name}_{names}.png")
    except:
        os.makedirs(path_save, exist_ok=True)
        plt.savefig(f"{path_save}/{name}_{names}.png")
    
    
def upload_theoretical_data(list_directory,K):
    dic_theo = {}
    for i in range(len(list_directory)):
        if i==0:
            try:
                error_quad = process_complex_file(list_directory[i][0])
                error_quad_sorted = error_quad[error_quad[:,0].argsort()]
                order_param_erf = process_complex_file(list_directory[i][1])
                order_param_erf_sorted = order_param_erf[order_param_erf[:,0].argsort()]
                alpha, q0, q1, m0, m1, v0, v1, free_energy, energy, train_error = order_param_erf_sorted[:,0], order_param_erf_sorted[:,1], order_param_erf_sorted[:,2],order_param_erf_sorted[:,3],order_param_erf_sorted[:,4],order_param_erf_sorted[:,5],order_param_erf_sorted[:,6], order_param_erf_sorted[:,7],order_param_erf_sorted[:,8],order_param_erf_sorted[:,9] 
                theo_normWs = q0 + v0 + (q1 + v1)/K
                theo_diag_ts = m0 + m1/K
                theo_off_ts = m1/K
                theo_off_ss = (q1+v1)/K
                dic_theo["errs_test"] = [error_quad_sorted]
                dic_theo["alpha"] = [alpha]
                dic_theo["normWs"] = [theo_normWs]
                dic_theo["DiagOverlapST"] = [theo_diag_ts]
                dic_theo["OffOverlapST"] = [theo_off_ts]
                dic_theo["OffOverlapSS"] = [theo_off_ss]
                dic_theo["errs"] = [train_error]
                dic_theo["losses"] = [energy]
                dic_theo["free_entropy"] = [free_energy]
            except:
                print(f"Something went wrong at i={i}")
        else:
            try:
                error_quad_1 = process_complex_file(list_directory[i][0])
                error_quad_1_sorted = error_quad_1[error_quad_1[:,0].argsort()]
                order_param_erf_1 = process_complex_file(list_directory[i][1])
                order_param_erf_1_sorted = order_param_erf_1[order_param_erf_1[:,0].argsort()]
                alpha_1, q0, q1, m0, m1, v0, v1, free_energy_1, energy_1, train_error_1 = order_param_erf_1_sorted[:,0], order_param_erf_1_sorted[:,1], order_param_erf_1_sorted[:,2],order_param_erf_1_sorted[:,3],order_param_erf_1_sorted[:,4],order_param_erf_1_sorted[:,5],order_param_erf_1_sorted[:,6], order_param_erf_1_sorted[:,7],order_param_erf_1_sorted[:,8],order_param_erf_1_sorted[:,9]
                theo_normWs_1 = q0 + v0 + (q1 + v1)/K
                theo_diag_ts_1 = m0 + m1/K
                theo_off_ts_1 = m1/K
                theo_off_ss_1 = (q1+v1)/K
                dic_theo["errs_test"].append(error_quad_1_sorted)
                dic_theo["alpha"].append(alpha_1)
                dic_theo["normWs"].append(theo_normWs_1) 
                dic_theo["DiagOverlapST"].append(theo_diag_ts_1)
                dic_theo["OffOverlapST"] = [theo_off_ts_1]
                dic_theo["OffOverlapSS"] = [theo_off_ss_1]
                dic_theo["errs"].append(train_error_1)
                dic_theo["losses"].append(energy_1)
                dic_theo["free_entropy"].append(free_energy_1)
            except:
                print(f"Something went wrong at i={i}")
    return dic_theo

def safe_save(data, filename):
    """
    Safely saves data to a .pt file in PyTorch, removing the backup after a successful save.
    
    Args:
        data: The data to save (e.g., a tensor or dictionary).
        filename (str): The target file name for saving.
    """
    # Step 1: Define paths for backup and temporary files
    backup_filename = f"{filename}.bak"
    temp_filename = f"{filename}.tmp"
    
    # Step 2: Save to a temporary file first
    try:
        torch.save(data, temp_filename)
        
        # Step 3: Rename the current file to backup if it exists
        if os.path.exists(filename):
            os.replace(filename, backup_filename)
        
        # Step 4: Rename the temporary file to the target filename
        os.replace(temp_filename, filename)
        #print(f"Data saved safely to {filename}.")
        
        # Step 5: Delete the backup file to save space
        if os.path.exists(backup_filename):
            os.remove(backup_filename)
            #print(f"Backup at {backup_filename} deleted to save space.")
    
    except Exception as e:
        # If something goes wrong, ensure no incomplete file replaces the original
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        #print(f"Error occurred during saving: {e}. Original file remains intact.")

    
def save_data_paper(keys, name, data,lr, names):
    dict = {}
    dict[keys] = {}
    try:
        for i in range(len(name)):
            dict[keys][name[i]] = data[name[i]]

        path_save_directory = f"data_for_paper"+names
        os.makedirs(path_save_directory, exist_ok=True)
        output_path = os.path.join(path_save_directory,f"{keys}_{lr}.pt")
        safe_save(dict, output_path) 
    except:
        dict[keys][name[0]] = data[name[0]]
        path_save_directory = f"data_for_paper"+names
        os.makedirs(path_save_directory, exist_ok=True)
        output_path = os.path.join(path_save_directory,f"{keys}_{lr}.pt")
        safe_save(dict, output_path) 
    #print("I did save the data pt 2")
    #torch.save(dict, output_path)
        
def safe_save_txt(data, filename):
    """
    Safely saves data as a NumPy array into a .txt file, preserving the original file 
    until the new data is saved successfully.

    Args:
        data: The data to save (e.g., a PyTorch tensor, list, or NumPy array).
        filename (str): The target file name for saving. Should end with `.txt`.
    """
    # Step 1: Convert data to a NumPy array
    if isinstance(data, torch.Tensor):
        data = data.numpy()  # Convert PyTorch tensor to NumPy
    elif isinstance(data, (list, tuple)):
        data = np.array(data)  # Convert list/tuple to NumPy
    elif isinstance(data, dict):
        raise ValueError("Cannot save a dictionary directly to a text file. Convert it first.")
    elif not isinstance(data, np.ndarray):
        raise ValueError("Unsupported data type. Please provide a NumPy array, tensor, or list.")
    
    # Step 2: Define paths for backup and temporary files
    backup_filename = f"{filename}.bak"
    temp_filename = f"{filename}.tmp"
    
    try:
        # Step 3: Save to a temporary file first
        np.savetxt(temp_filename, data)
        
        # Step 4: Rename the current file to backup if it exists
        if os.path.exists(filename):
            os.replace(filename, backup_filename)
        
        # Step 5: Rename the temporary file to the target filename
        os.replace(temp_filename, filename)
        
        # Step 6: Delete the backup file to save space
        if os.path.exists(backup_filename):
            os.remove(backup_filename)
    
    except Exception as e:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        # Re-raise the exception for debugging or logging
        raise RuntimeError(f"Failed to save data to {filename}: {e}")

    
    
    

    
    
    