from TreeLearner import CustomDecisionTree
#from new_ex import initial_dist_data, initial_valid_data, second_phase_dist_data, second_phase_valid_data
import pandas as pd
import numpy as np
import copy
import random
import time
import os

PATH = os.path.realpath("")


def get_split_count_map(tree):
    split_count_map = {}
    tree_nodes = tree.get_tree_snapshot()
    for idx, (parent_feature, node_feature, node_id) in enumerate(tree_nodes):
        if node_feature == None:
            continue
        if node_feature not in split_count_map:
            split_count_map[(parent_feature, node_feature, node_id)] = 1
    return split_count_map


def update_split_count_map(tree, split_count_map):
    tree_nodes = tree.get_tree_snapshot()
    nodes_in_map_not_in_tree = [key for key in split_count_map.keys() if key not in tree_nodes and key[0] != None]
    tree_nodes_search = [(node_feature, node_id) for (_, node_feature, node_id) in tree_nodes]
    
    for idx, (parent_feature, node_feature, node_id) in enumerate(nodes_in_map_not_in_tree):
        if (parent_feature, int(node_id / 10)) in tree_nodes_search:
            continue
        else:
            del split_count_map[(parent_feature, node_feature, node_id)]
            #print(f"Removed an element from the split count map: {(parent_feature, node_feature, node_id)}")
    new_tree_split_count_map = get_split_count_map(tree)
    for idx, (parent_feature, node_feature, node_id) in enumerate(new_tree_split_count_map.keys()):
        if (parent_feature, node_feature, node_id) not in split_count_map:
            split_count_map[(parent_feature, node_feature, node_id)] = 1
            #print(f"Added an element to the split count map: {(parent_feature, node_feature, node_id)}")
    return split_count_map
    



def get_ratio(tree_copy, mutation, leaf, distCurr, validCurr, split_count_map):
    max_split_count = 10000
    if mutation == 'split':
        try:
            '''If trying to split a leaf node that has only one datapoint, it will raise a ValueError.
            In this case, we would like to discard this move and try another one.'''
            pfeature, nfeature = tree_copy.split_node(leaf)
            if (pfeature, nfeature, leaf) in split_count_map:
                if split_count_map[(pfeature, nfeature, leaf)] >= max_split_count:
                    raise ValueError
                else:
                    split_count_map[(pfeature, nfeature, leaf)] += 1
            else:
                split_count_map[(pfeature, nfeature, leaf)] = 1
        except ValueError:
            return None
        error_bounds = tree_copy.get_error_bounds()
        gaindist = distCurr - error_bounds['dist_error']
        gainvalid = validCurr - error_bounds['valid_error']
        if gaindist + gainvalid < 0:
            return None
        move = ((gaindist+gainvalid)/2, tree_copy, error_bounds, leaf, 'split')
        return move
    elif mutation == 'prune':
        try:
            '''If trying to prune the root node that is the only node in the tree, it will raise a ValueError.'''
            tree_copy.prune_node(leaf)
        except ValueError:
            return None
        error_bounds = tree_copy.get_error_bounds()
        gaindist = distCurr - error_bounds['dist_error']
        gainvalid = validCurr - error_bounds['valid_error']
        if gaindist + gainvalid < 0:
            return None
        move = ((gaindist+gainvalid), tree_copy, error_bounds, leaf, 'prune')
        split_count_map = update_split_count_map(tree_copy, split_count_map)
        return move




def Normalize(moves):
    '''I want to perform weighted normalization here'''
    total = sum([move[0] for move in moves])
    if total == 0:
        return [(1, move[1], move[2], move[3], move[4]) for move in moves]
    moves = [(move[0]/total, move[1], move[2],move[3], move[4]) for move in moves]
    return moves



#MuteTree(tree, dataset, distCurr, validCurr)
def MuteTree(tree, df, distUser, validUser):
    start = time.time()
    tree.predict(df[list(df.columns[:-3])].values, expected_labels=df['label'].values, weights=df['weight'].values, member=df['member'].values)
    error_bounds = tree.get_error_bounds()
    print("Second Phase Bounds:", error_bounds)
    tree.save_tree(f'{PATH}/ex11_after_second_phase')
    distCurr, validCurr = error_bounds['dist_error'], error_bounds['valid_error']
    set_of_mutations = ['split', 'prune']
    split_count_map = get_split_count_map(tree)
    tree_sequence = {}
    while(time.time() - start < 200):
        print('{error_bounds=}', tree.get_error_bounds())
        leaf_ids = tree.get_all_leaf_nodes()
        #topk = round(len(clf.get_all_leaf_nodes())/2) + 1 
        moves = []
        for mutation in set_of_mutations:
            for leaf in leaf_ids:
                tree_copy = copy.deepcopy(tree)
                move = get_ratio(tree_copy, mutation, leaf, distCurr, validCurr, split_count_map)
                if move:
                    moves.append(move)
        #print("All moves:", [(move[0], move[3], move[4]) for move in moves])
        if not moves:
            raise ValueError("No valid moves found. Try increasing the max_split_count.")
        moves = Normalize(moves)
        moves.sort(key=lambda x: x[0], reverse=True)
        _, tree, error_bounds, leaf_id, mutation = random.choices(moves, weights=[move[0] for move in moves])[0]
        #print("leaf_id and mutation", leaf_id, mutation)
        tree.save_tree(f'{PATH}/intermediate/ex5_{leaf_id}_{mutation}')
        tree_sequence[tree] = error_bounds
        distCurr, validCurr = error_bounds['dist_error'], error_bounds['valid_error']
        if distCurr <= distUser  and validCurr <= validUser:
            #print(f'{split_count_map=}')
            print("Successfully reached the desired user error bounds.")
            return tree, tree_sequence
    '''If timeout occurs, we return the tree with the best error bounds so far by sorting the dictionary tree_sequence and extracting the first element.'''
    print("Timeout occurred. Returning the best tree so far.")
    #print(tree.print_leaf_datapoints())
    # best_tree = sorted(tree_sequence.items(), key=lambda x: round(np.sqrt(x[1]['dist_error'] + x[1]['valid_error'])))[0][0]
    best_tree = sorted(tree_sequence.items(), key=lambda x: round(np.sqrt(x[1]['dist_error'] + x[1]['valid_error']), 2))[0][0]
    print("Best Tree Error Bounds:", best_tree.get_error_bounds())
    return best_tree, tree_sequence

from sympy.logic.boolalg import Or, And, Not
from sympy import symbols, simplify_logic
import re

def parse_dnf(dnf_formula):
    """
    Parse a DNF formula string into a sympy Boolean expression.
    Args:
        dnf_formula (str): DNF formula as a string, e.g., "(x1 && x2 && x3) || (x3 && !x4) || (!x1 && x2)"
    Returns:
        sympy Boolean expression representing the DNF formula.
    """
    # Replace '&&' with '&' and '||' with '|' for Python's logical operators
    dnf_formula = dnf_formula.replace('&&', '&').replace('||', '|').replace('!', '~').replace('(','').replace(')','')

    # Extract all unique variables using regex (handles negated variables too)
    variables = set(re.findall(r'\b\w+\b', dnf_formula))
    sympy_vars = {var: symbols(var) for var in variables}

    # Prepare evaluation dictionary
    eval_globals = {"__builtins__": None}
    eval_globals.update(sympy_vars)
    eval_globals.update({"&": And, "|": Or, "~": Not})

    # Evaluate the formula with `eval`
    expr = eval(dnf_formula, eval_globals)
    return expr

def DNF_to_CNF(dnf_formula):
    """
    Convert a DNF formula string to a minimized CNF formula string.
    Args:
        dnf_formula (str): DNF formula as a string, e.g., "(x1 && x2 && x3) || (x3 && !x4) || (!x1 && x2)"
    Returns:
        cnf_formula (str): Minimized CNF formula as a string, e.g., "(!x1 || x3) && (!x1 || !x4) && x2"
    """
    # Step 1: Parse the DNF formula into a sympy expression
    dnf_expr = parse_dnf(dnf_formula)
    print(dnf_expr)
    # Step 2: Convert to CNF and simplify
    cnf_expr = simplify_logic(dnf_expr, form='cnf')
    print(cnf_expr)
    # Step 3: Convert sympy expression back to the desired string format
    cnf_str = str(cnf_expr)

    # Replace sympy's logical operators with the string format
    cnf_str = cnf_str.replace('Or', '||').replace('And', '&&').replace('~', '!')

    # Ensure that '&&' and '||' are properly spaced for clarity
    cnf_str = cnf_str.replace('&&', ') && (').replace('||', ' || ').strip()
    cnf_str = cnf_str.replace('&','&&').replace('|','||')
    return cnf_str


def decimal_to_n_bit_binary(decimal, n):
    """Convert a decimal number to n-bit binary as a list of integers."""
    return list(map(int, format(decimal, f'0{n}b')))

def prepare_data(data, number_of_vars, DistEstimate = True):
    """Convert a list of tuples into a DataFrame with binary representations, label, weight, and member."""
    records = []
    member = 1 if DistEstimate else 0
    for idx, (decimal, weight, label) in enumerate(data):
        var_binary = decimal_to_n_bit_binary(decimal, number_of_vars)
        records.append(var_binary + [label, weight, member])
    columns = [f'x{i+1}' for i in range(number_of_vars)] + ['label', 'weight', 'member']
    df = pd.DataFrame(records, columns=columns)
    return df


initial_dist_data = [(0, 1, 1), (1, 1, 1), (6, 1, 1), (7, 1, 1), (24, 1, 1), (25, 1, 1), (30, 1, 1), (31, 1, 1)]
initial_valid_data = [(5, 1, 0), (17, 1, 0), (19, 1, 0), (20, 1, 0), (21, 1, 0), (23, 1, 0), (26, 1, 0), (28, 1, 0)]
second_phase_dist_data = [(16, 1, 1), (28, 1, 1), (26, 1, 1), (18, 1, 1)]
second_phase_valid_data = [(2, 1, 0), (3, 1, 0), (4, 1, 0), (8, 1, 0), (9, 1, 0), (10, 1, 0), (11, 1, 0), (14, 1, 0), (15, 1, 0), (27, 1, 0), (29, 1, 0)]
third_phase_dist_data = [(12,1,1)]
third_phase_valid_data = [(17,1,0),(20,1,0),(22,1,0),(23,1,0)]
fourth_phase_dist_data = []
fourth_phase_valid_data = [(13, 1, 0)]

number_of_vars = 5
df_dist = prepare_data(initial_dist_data, number_of_vars, DistEstimate = True)
df_valid = prepare_data(initial_valid_data, number_of_vars, DistEstimate = False)
df = pd.concat([df_dist, df_valid], ignore_index=True)
X = df[list(df.columns[:-3])].values
y = df['label'].values
clf = CustomDecisionTree(max_depth=8)
clf.fit_initial(X, y, feature_names=df.columns[:-3])
#print("initial dnf", clf.tree_to_dnf())
clf.save_tree(f'{PATH}/ex11_before_prediction')
predictions = clf.predict(df_dist[list(df.columns[:-3])].values, expected_labels=df_dist['label'].values, weights=df_dist['weight'].values, member=df_dist['member'].values)
#print("Initial Bounds(after prediction):", clf.get_error_bounds())
clf.save_tree(f'{PATH}/ex11_after_prediction')
#print("dnf", clf.tree_to_dnf())
df_dist_second = prepare_data(second_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_second = prepare_data(second_phase_valid_data, number_of_vars, DistEstimate = False)
df_second = pd.concat([df_dist_second, df_valid_second], ignore_index=True)
clf.predict(df_second[list(df.columns[:-3])].values, expected_labels=df_second['label'].values, weights=df_second['weight'].values, member=df_second['member'].values)

# df_dist_third = prepare_data(third_phase_dist_data, number_of_vars, DistEstimate = True)
# df_valid_third = prepare_data(third_phase_valid_data, number_of_vars, DistEstimate = False)
# df_third = pd.concat([df_dist_third, df_valid_third], ignore_index=True)

# df_dist_fourth = prepare_data(fourth_phase_dist_data, number_of_vars, DistEstimate = True)
# df_valid_fourth = prepare_data(fourth_phase_valid_data, number_of_vars, DistEstimate = False)
# df_fourth = pd.concat([df_dist_fourth, df_valid_fourth], ignore_index=True)

for iteration in range(4,5):
    clf_copy = copy.deepcopy(clf)
    print(f'{iteration=}')
    random.seed(iteration + 10)
    print("Second Phase")
    final_tree, tree_sequence = MuteTree(clf_copy, df_second, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex5_second_{iteration}')
    print("CNF:", DNF_to_CNF(final_tree.tree_to_dnf()))



