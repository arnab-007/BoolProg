import os 
import json
import random


CURRENT_PATH = os.path.realpath("")
DIRECTORY_PATH = os.path.dirname(CURRENT_PATH)
PATH = os.path.dirname(DIRECTORY_PATH)
#print(PATH)
assumed_shape = " "
with open(os.path.join(PATH, "program-list.txt"), "r") as f:
    prognames = f.read().strip().split("\n")

"""
[get_config] loads the json object in [progname]'s configuration file
"""
def get_config(progname):
    with open(os.path.join(PATH, "candidate_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config


def parse_dnf(dnf_string):
    """
    Parse a DNF string into a list of terms, where each term is a list of literals.
    
    Args:
        dnf_string (str): The DNF formula string.
        
    Returns:
        list: A list of terms, where each term is a list of literals.
    """
    # Split by '||' to get terms
    terms = dnf_string.split("||")
    # Split each term by '&&' to get literals
    parsed_terms = [term.strip("() ").split("&&") for term in terms]
    return parsed_terms

def get_variables_from_literal(literal):
    """
    Extract the variable name from a literal.

    Args:
        literal (str): A literal in the form of 'x1' or '!x2'.

    Returns:
        str: The variable name (e.g., 'x1' for 'x1' or '!x1').
    """
    return literal.strip("!")


def get_unique_dict(input_dict):
    """
    Extract unique variables from the dictionary, ignoring whitespace and '!' in keys.

    Args:
        input_dict (dict): Input dictionary with potentially duplicate and malformed keys.

    Returns:
        dict: Dictionary with unique variables and their values.
    """
    unique_dict = {}
    
    for key, value in input_dict.items():
        # Normalize the variable name by stripping spaces and removing '!'
        normalized_key = key.strip().lstrip('!')
        
        # Add to the dictionary if the normalized key isn't already present
        if normalized_key not in unique_dict:
            unique_dict[normalized_key] = value


    terms = []
    for var, value in unique_dict.items():
        if value == 0:
            terms.append(f"!{var}")
        else:
            terms.append(f"{var}")
    return " && ".join(terms)

def DNF_sampler(dnf_string, variables):
    """
    Sample an assignment from a DNF string almost uniformly.
    
    Args:
        dnf_string (str): The DNF formula as a string.
        variables (list): List of all variables in the DNF formula.
        
    Returns:
        dict: A dictionary of sampled values for all variables.
    """
    # Parse the DNF string into terms
    term_list = parse_dnf(dnf_string)
    
    # Choose a term uniformly at random
    chosen_term = random.choice(term_list)
    
    # Determine the support variables of the chosen term
    support_vars = set(get_variables_from_literal(literal) for literal in chosen_term)
    support_vars = {key.strip().lstrip('!') for key in support_vars}
    # Determine the non-support variables
    non_support_vars = [var for var in variables if var not in support_vars]
    #print(non_support_vars)

    
    # Assign random values to non-support variables
    sampled_assignment = {var: random.randint(0, 1) for var in non_support_vars}
    #print(sampled_assignment)
    chosen_term = [var.strip() for var in chosen_term]
    # Assign fixed values to support variables based on the term
    for literal in chosen_term:
        var = get_variables_from_literal(literal)
        sampled_assignment[var] = 0 if literal.startswith("!") else 1
    
    return sampled_assignment




def nnf_to_dimacs(cnf_formula, variable_mapping):
    dimacs = []
    for clause in cnf_formula:
        dimacs_clause = []
        for literal in clause:
            if literal.startswith('!'):
                dimacs_clause.append(-variable_mapping[literal[1:]])
            else:
                dimacs_clause.append(variable_mapping[literal])
        dimacs.append(dimacs_clause)
    return dimacs






results = {}
for progname in prognames:
    config = get_config(progname)
    prog_variables = config["Program_variables"]["Bools"]
    #print(prog_variables)
    cand = config["Candidate"]["Expression"]
    init_states = config["Initial states"]["Expression"]


init_list_clauses = init_states.split('&&')
print(init_list_clauses)
init_list = list()
for clause in init_list_clauses:

    literals = clause.split('||')
    literals = [literal.strip("() ") for literal in literals]
    # Filter out empty strings
    literals = list(filter(None, literals))
    init_list.append(literals)

#print(init_list)
    
cand_list_terms = cand.split('||')
print(cand_list_terms)
cand_list = list()
for term in cand_list_terms:

    literals = term.split('&&')
    literals = [literal.strip("() ") for literal in literals]
    # Filter out empty strings
    literals = list(filter(None, literals))
    cand_list.append(literals)

#print(cand_list)



# Example CNF formula
dnf_formula_cand = cand_list
cnf_formula_init = init_list
# Variable mapping (manual assignment of variables to integers)
#variable_mapping = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4, 'x5': 5}
#variable_mapping = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4, 'x5': 5, 'x6': 6, 'x7': 7, 'x8': 8}
#variable_mapping = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4, 'x5': 5, 'x6': 6, 'x7': 7, 'x8': 8, 'x9': 9}
variable_mapping = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4, 'x5': 5, 'x6': 6, 'x7': 7, 'x8': 8, 'x9': 9, 'x10': 10}
#variable_mapping = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4, 'x5': 5, 'x6': 6, 'x7': 7, 'x8': 8, 'x9': 9, 'x10': 10, 'x11': 11, 'x12': 12, 'x13': 13, 'x14': 14, 'x15': 15, 'x16': 16, 'x17': 17}
#variable_mapping = {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4, 'x5': 5, 'x6': 6, 'x7': 7, 'x8': 8, 'x9': 9, 'x10': 10, 'x11': 11, 'x12': 12}
# Convert to DIMACS format
dimacs_cand = nnf_to_dimacs(dnf_formula_cand, variable_mapping)
dimacs_init = nnf_to_dimacs(cnf_formula_init, variable_mapping)
#print(dimacs_init)

num_vars = len(variable_mapping)  # Number of variables
print(num_vars)
def write_cnf_dimacs_to_file(dimacs, num_vars, num_clauses, file_name):
    with open(file_name, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in dimacs:
            f.write(" ".join(map(str, clause)) + " 0\n")

def write_dnf_dimacs_to_file(dimacs, num_vars, num_terms, file_name):
    with open(file_name, 'w') as f:
        f.write(f"p dnf {num_vars} {num_terms}\n")
        for term in dimacs:
            f.write(" ".join(map(str, term)) + " 0\n")

# Write DIMACS formula to file


write_dnf_dimacs_to_file(dimacs_cand, num_vars, len(dimacs_cand), os.path.join(PATH, "candidate-dnf"))
#write_dimacs_to_file(dimacs_cand, num_vars, len(dimacs_cand), 'candidate-cnf')

write_cnf_dimacs_to_file(dimacs_init, num_vars, len(dimacs_init), os.path.join(PATH, "input-cnf"))

'''

import subprocess

# Path to your bash script
script_path = os.path.join(PATH, "src/Validifier/cmsgen-sampler.sh") 
#print("script_path",script_path)
# Run the bash script
result = subprocess.run(["bash", script_path],capture_output=True,text=True,check=True)  # Raise CalledProcessError if the subprocess fails

'''       

#print(result)
# Define the variable mapping
#variable_mapping = {1: 'x1', 2: 'x2', 3: 'x3', 4: 'x4', 5: 'x5'}
#variable_mapping = {1: 'x1', 2: 'x2', 3: 'x3', 4: 'x4', 5: 'x5', 6: 'x6', 7: 'x7', 8: 'x8'}
variable_mapping = {1: 'x1', 2: 'x2', 3: 'x3', 4: 'x4', 5: 'x5', 6: 'x6', 7: 'x7', 8: 'x8', 9: 'x9', 10: 'x10'}
#variable_mapping = {1: 'x1', 2: 'x2', 3: 'x3', 4: 'x4', 5: 'x5', 6: 'x6', 7: 'x7', 8: 'x8', 9: 'x9', 10: 'x10', 11: 'x11', 12: 'x12', 13: 'x13', 14: 'x14', 15: 'x15', 16: 'x16', 17: 'x17'}
#variable_mapping = {1: 'x1', 2: 'x2', 3: 'x3', 4: 'x4', 5: 'x5', 6: 'x6', 7: 'x7', 8: 'x8', 9: 'x9', 10: 'x10', 11: 'x11', 12: 'x12'}




def convert_sample(sample_line):
    # Split the line into tokens
    tokens = sample_line.split()
    # Initialize an empty list to hold the formatted variables
    formatted_vars = []
    
    for token in tokens:
        num = int(token)
        if num != 0:
            if num > 0:
                var_name = variable_mapping.get(num)
                if var_name:
                    formatted_vars.append(f"{var_name}")
            else:
                var_name = variable_mapping.get(-num)
                if var_name:
                    formatted_vars.append(f"!{var_name}")
    
    return " && ".join(formatted_vars)

'''
# Read the input file and convert the content
with open('samples.out', 'r') as file:
    lines = file.readlines()

converted_lines = [convert_sample(line) for line in lines]

'''

candidate_samples = []
for _ in range(500):
  #print("Sampled Assignment:")
  sampled_assignment = get_unique_dict(DNF_sampler(cand, prog_variables))
  candidate_samples.append(sampled_assignment)
  #print(sampled_assignment)

# Write the converted content to a new file
with open('samples_converted.txt', 'w') as file:
    file.write("\n".join(candidate_samples))




