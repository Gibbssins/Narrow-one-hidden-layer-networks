# -*- coding: utf-8 -*-

from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F 
import re
import os
from scipy.optimize import linear_sum_assignment

import torch
import time
import pynvml
import threading

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)



def check_diff_num(func, dfuncs, epss, num_checks, wrap = False, par = None, zero_out_var = None):
    
    n_var = len(dfuncs)

    dvars = np.zeros((num_checks, n_var))
    dvars_num = np.zeros((num_checks, n_var, len(epss)))

    I = np.eye(n_var)

    for c in range(num_checks):

        var = np.random.rand(n_var)
        if zero_out_var is not None:
            var[zero_out_var] = 0.

        for v, dfunc in enumerate(dfuncs):
            if par is not None:
                if wrap:
                    dvars[c,v] = dfunc(var, par)
                else:
                    dvars[c,v] = dfunc(*var, *par)
            else:
                if wrap:
                    dvars[c,v] = dfunc(var)
                else:
                    dvars[c,v] = dfunc(*var)

        for v in range(n_var):
            for iep, eps in enumerate(epss):
                var_l = var - I[v] * 0.5 * eps
                var_r = var + I[v] * 0.5 * eps
                if par is not None:
                    if wrap:
                        f_l = func(var_l, par)
                        f_r = func(var_r, par)
                    else:
                        f_l = func(*var_l, *par)
                        f_r = func(*var_r, *par)
                else:
                    if wrap:
                        f_l = func(var_l)
                        f_r = func(var_r)
                    else:
                        f_l = func(*var_l)
                        f_r = func(*var_r)
                        
                dvars_num[c,v,iep] = (f_r - f_l) / eps
                
    return dvars, dvars_num


def check_equal(arr1, arr2, name, verbose = False):
    if not np.allclose(arr1, arr2, equal_nan=True):
        print(f"Something wrong with {name}!!")
    else:
        if verbose:
            print(f"All good with {name} (except NaNs)!")
        else:
            pass
        

def plot_compare(v1, v2):
    plt.plot(v1.flatten(), v1.flatten(), ':')
    plt.plot(v1.flatten(), v2.flatten(), '.');


def extract_batch(data, target,
                  set_on_device = True,
                  reshape_data = False,
                  device = torch.device("cpu")):

    if not set_on_device:
        data, target = data.to(device), target.to(device)
    if reshape_data:
        data = data.reshape(data.shape[0], data.shape[-2] * data.shape[-1])
        
    return data, target


def get_dict(net, to_numpy=False):
    if to_numpy:
        return OrderedDict(
            {k: v.detach().clone().to("cpu").numpy() for k, v in net.state_dict().items()}
        )
    else:
        return OrderedDict(
            {k: v.detach().clone().to("cpu") for k, v in net.state_dict().items()}
        )
    

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_singular(A):
    return np.linalg.matrix_rank(A) < len(A)


def log(msg, logfile, print_to_out=True):
    """
    Print log message to  stdout and the given logfile.
    """
    logfile.write(msg + "\n")

    if print_to_out:
        print(msg)
        
        
def minmax(tensor, tmin, tmax):
    return torch.min(torch.max(tensor.data, tmin, out=tensor.data), tmax, out=tensor.data)


def argmin_mat(matrix):
    lin_idx_min = np.nanargmin(matrix)
    idx_min = np.unravel_index(lin_idx_min, matrix.shape)
    return idx_min 



def rank_column_highest_on_diagonal(matrix):
    # Convert the matrix to a PyTorch tensor for easier manipulation
    matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Get the number of rows (and columns, assuming the matrix is square)
    n = matrix.size(0)
    
    # To keep track of the current column arrangement
    current_columns = list(range(n))
    current_max_columns = [max(matrix[:,i]) for i in range(n)]
    column_swap_count = [0. for i in range(n)]
    
    for i in range(n):
        # Find the index of the maximum element in the i-th row
        max_index = torch.argmax(matrix[i]).item()
        max_el = torch.max(matrix[:,max_index])
        
        
        # Check if we need to move the column
        if current_columns[i] != max_index:
            # Get the current value at the diagonal
            current_diagonal_value = matrix[i, i].item()
            new_diagonal_value = matrix[i, max_index].item()

            # Only move the column if the new value is greater than the current diagonal value
            if new_diagonal_value > current_diagonal_value:
                # Swap columns i and max_index
                if column_swap_count[max_index] == 0:
                    matrix[:, [i, max_index]] = matrix[:, [max_index, i]]
                    # Update the current_columns list to reflect the swap
                    current_columns[i], current_columns[max_index] = current_columns[max_index], current_columns[i]
                    column_swap_count[i] +=1
                    #print(column_swap_count, i)
                else:
                    if new_diagonal_value >= max_el:
                        matrix[:, [i, max_index]] = matrix[:, [max_index, i]]
                        # Update the current_columns list to reflect the swap
                        current_columns[i], current_columns[max_index] = current_columns[max_index], current_columns[i]
                        column_swap_count[current_columns[i]] +=1
                             

    return matrix 


def rank_highest_on_diagonal(matrix):
    # Convert the matrix to a PyTorch tensor for easier manipulation
    matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Get the number of rows (and columns, assuming the matrix is square)
    n = matrix.size(0)
    
    for i in range(n):
        # Find the index of the maximum element in the i-th row
        max_index = torch.argmax(matrix[i]).item()
        
        # Swap the element at the diagonal position with the maximum element in the i-th row
        if max_index != i:
            # Temporarily store the value at the diagonal
            temp = matrix[i, i].item()
            # Move the highest element to the diagonal position
            matrix[i, i] = matrix[i, max_index]
            # Move the temporarily stored value to the position of the highest element
            matrix[i, max_index] = temp
    
    return matrix 

def extract_alpha(string):
    match = re.search(r'_a(\d+\.\d+)_', string)
    if match:
        return float(match.group(1))
    else:
        return None 

def load_files_as_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def compare_pt_files(file1, file2):
    # Load the PyTorch models or tensors from both files
    model1 = torch.load(file1)
    model2 = torch.load(file2)
    
    # Check if the type of both objects is the same
    if type(model1) != type(model2):
        #print("The two files contain different types of objects.")
        return False
        
    # Compare the models or tensors
    if isinstance(model1, dict):
        # For models saved as state_dict (dictionaries)
        for key in model1:
            if key not in model2 or not torch.equal(model1[key], model2[key]):
                #print(f"Difference found at key: {key}")
                return False
    else:
        # For models saved directly as tensors
        if not torch.equal(model1, model2):
            #print("The two tensors/models are not identical.")
            return False
    
    #print("The two .pt files are identical.")
    return True


def compare_dicts(dict1, dict2):
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        print("Dictionaries have different keys!")
        return False

    # Iterate over each key and compare the tensors
    for key in dict1:
        tensor1 = dict1[key]
        tensor2 = dict2[key]
        
        if not torch.equal(tensor1, tensor2):
            # If exact comparison fails, check approximate similarity
            if torch.allclose(tensor1, tensor2, rtol=1e-08, atol=1e-015):
                print(f"Tensors at key '{key}' are approximately equal.")
            else:
                print(f"Tensors at key '{key}' are not equal.")
                return False
        else:
            print(f"Tensors at key '{key}' are exactly equal.")
    
    return True



def get_all_parameters(net, to_numpy=False):
    
    params_dict = {}
    list_name_params = []
    for name, param in net.named_parameters(recurse=True):
        list_name_params.append(f"{name}")
        if to_numpy:
            # Convert to numpy if specified
            params_dict[f"{name}"] = param.detach().clone().to("cpu")
        else:
            params_dict[f"{name}"] = param
    if not "v" in list_name_params:
        params_dict["v"] = net.v.detach().clone().to("cpu")

    for name, buffer in net.named_buffers(recurse=True):
        # Include non-learnable parameters (buffers) like running_mean or running_var in BatchNorm layers
        if to_numpy:
            params_dict[f"{name}"] = buffer.detach().clone().to("cpu")
        else:
            params_dict[f"{name}"] = buffer

    return params_dict

def check_string(input_string):
    return input_string.strip().lower() == "true" 


### Save directory
def append_unique_line(file_path, new_line):
    # Read the existing lines from the file
    if not os.path.exists(file_path):
        # Create the file and add the new line
        with open(file_path, 'w') as file:
            file.write(new_line + '\n')
    else:
        # Read the existing lines from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Check if the new line is already in the file
        if new_line + '\n' not in lines:
            # If not, append the new line to the list of lines
            lines.append(new_line + '\n')

            # Write the updated lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
                
### thetha loss
def theta_error(y1, y2, margin=0):
    return torch.clamp(margin - y1 * y2, min=0)

def hinge_loss(y1, y2, margin=0):
    return (torch.sign(y1)*y1)[torch.sign(y1) * torch.sign(y2)<=0]

"""def hinge_loss(y1, y2, margin=0):
    return torch.clamp(margin - y1 * y2, min=0)"""

"""def hinge_loss(output, y, margin=0.0):
    return (output[output * y <= margin]).sum()"""
    
"""def hinge_loss(y1, y2, margin=1):
    return (y1 - y2)**2"""



"""def select_loss_function(name):
    #Selects a loss function based on the given name.
    loss_functions = {
        "mse": F.mse_loss,
        "hinge": lambda output, target: torch.mean(torch.clamp(1 - output * target, min=0)),
        "theta": theta_loss
    }
    
    return loss_functions.get(name.lower(), None)
"""



"""def hinge_loss(y1, y2, margin=0):
    return (torch.sign(y1)*y1)[torch.sign(y1) * torch.sign(y2)<=0]"""


             
                
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
        
        
def sort_indicesold(array, stu_array):
    """
    Sorts indices of `stu_array` based on the correlation with `array` 
    and returns a new array sorted accordingly.

    Parameters:
    array (torch.Tensor): The input tensor (e.g., teacher data), shape (K, N).
    stu_array (torch.Tensor): The student tensor to sort, shape (M, N).

    Returns:
    torch.Tensor: The sorted student tensor, shape same as stu_array.
    """
    # Ensure tensors are on the same device
    device = array.device
    stu_array = stu_array.to(device)
    
    # Compute correlation table
    correlation_table = array @ stu_array.T  # Shape: (K, M)
    K, M = correlation_table.shape
    factor = int(M//K)
    new_stu_array = torch.empty_like(stu_array)
    p = 0
    
    for row in range(K):
        # Get indices of the top-K correlations for the row
        indices = torch.argsort(correlation_table[row], descending=True)[:K]
        # Assign the sorted rows to the new student array
        for j, index in enumerate(indices):
            if p + j * K < factor * K:
                new_stu_array[p + j * K] = stu_array[index]
            else:
                break
        p += 1
    
    return new_stu_array



def sort_indices(array, stu_array):
    device = array.device
    stu_array = stu_array.to(device)
    
    # Compute correlation matrix
    correlation = array @ stu_array.T  # Shape: (K, M)
    K, M = correlation.shape
    
    if K == M:
        # Use Hungarian algorithm for optimal assignment
        cost = -correlation.cpu().numpy()  # Minimization problem
        row_ind, col_ind = linear_sum_assignment(cost)
        new_stu_array = stu_array[col_ind].clone()
    else:
        # Handle M > K: Assign top matches and append the rest
        new_stu_array = []
        assigned = set()
        # Assign best match for each row in array
        for i in range(K):
            idx = torch.argmax(correlation[i]).item()
            new_stu_array.append(stu_array[idx])
            assigned.add(idx)
        # Append remaining rows not matched
        for j in range(M):
            if j not in assigned:
                new_stu_array.append(stu_array[j])
        new_stu_array = torch.stack(new_stu_array)
    
    return new_stu_array
