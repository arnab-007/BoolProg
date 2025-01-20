import subprocess
import random
import re
import os
import json
import sys
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

CURRENT_PATH = os.path.realpath("")
PATH = os.path.dirname(parent_dir)
assumed_shape = " "



"""
[get_config] loads the json object in [progname]'s configuration file
"""


def get_config(progname):
    with open(os.path.join(PATH, "candidate_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config



with open(os.path.join(PATH, "program-list.txt"), "r") as f:
    prognames = f.read().strip().split("\n")

for progname in prognames:
    config = get_config(progname)
    prog_variables = config["Program_variables"]["Bools"]
    cand = config["Candidate"]["Expression"]
    init_states = config["Initial states"]["Expression"]
    k = config["Program specification"]["iterations"]


 

import random

def balance_lists(list1, list2):
    """
    Balance the number of elements in two lists by sampling uniformly at random from the larger list.
    
    Args:
    list1 (list): The first list (smaller or equal in size to list2).
    list2 (list): The second list (larger list from which to sample).
    
    Returns:
    tuple: A tuple containing list1 and the sampled version of list2 with the same length as list1.
    """
    if len(list1) > len(list2):
        raise ValueError("list1 must be smaller or equal in size to list2.")
    
    sampled_list2 = random.sample(list2, len(list1))
    return list1, sampled_list2


def sample_to_decimal(sample):
    
    #binary_string = ['0'] * 17
    #binary_string = ['0'] * 8
    #binary_string = ['0'] * 5
    #binary_string = ['0'] * 9
    binary_string = ['0'] * 10
    #binary_string = ['0'] * 12
    #print(sample)
    for value in sample:
        if value == 0:
            continue  # Skip the terminating zero
        index = abs(value) - 1  # Convert variable to zero-based index
        if value > 0:
            binary_string[index] = '1'  # Set bit to 1 for positive values
        else:
            binary_string[index] = '0'  # Set bit to 0 for negative values
    
    # Join the bits to form a binary string and convert to decimal
    binary_str = ''.join(binary_string)
    return int(binary_str, 2)

# Read samples from the file and convert each sample to decimal
decimals_1 = []
with open('samples_plus.out', 'r') as f:
    for line in f:
        # Convert the line into a list of integers
        sample = list((map(int, line.split())))
        decimal_value = sample_to_decimal(sample)
        decimals_1.append((decimal_value))

decimals_2 = []
with open('samples_minus.out', 'r') as f:
    for line in f:
        # Convert the line into a list of integers
        sample = list((map(int, line.split())))
        decimal_value = sample_to_decimal(sample)
        decimals_2.append((decimal_value))


decimals_1 = sorted(list(set(decimals_1)))
decimals_2 = sorted(list(set(decimals_2)))
decimals_S, decimals_not_S = balance_lists(decimals_1, decimals_2)
decimals_S = sorted(list(set(decimals_S)))
decimals_not_S = sorted(list(set(decimals_not_S)))
#print(len(decimals_S))
#print(len(decimals_not_S))
print("Initial states in S",decimals_S)
print("Initial states not in S",decimals_not_S)



input_dict = {"progname":progname,"candidate":cand,"init_states":init_states,"iterations":k}
output_dict = {"Sampled positive initial states":decimals_S,"Sampled negative initial states":decimals_not_S}
total_dict = {"input_dict":input_dict,"output_dict":output_dict}


results_directory = os.path.join(CURRENT_PATH, 'Sampler_results')
if not os.path.exists(results_directory):
    os.makedirs(results_directory)



existing_files = os.listdir(results_directory)
file_number = 1

'''
# Loop until you find an unused file name (exp1.json, exp2.json, etc.)
while f"init_{progname}_{file_number}.json" in existing_files:
    file_number += 1
'''
# Create the new filename in the results subdirectory
filename = os.path.join(results_directory, f"init_{progname}_{file_number}.json")

# Write the data to the new file
with open(filename, 'w') as json_file:
    json.dump(total_dict, json_file, indent=4)




