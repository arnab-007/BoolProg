from TreeLearner import CustomDecisionTree
#from new_ex import initial_dist_data, initial_valid_data, second_phase_dist_data, second_phase_valid_data
import pandas as pd
import numpy as np
import copy
import random
import time
import os
import sys
import json
import subprocess

PATH = os.path.realpath("")
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
PARENT_PATH = os.path.dirname(parent_dir)
assumed_shape = " "



"""
[get_config] loads the json object in [progname]'s configuration file
"""


def get_init_state(progname):
    with open(os.path.join(PARENT_PATH, "src/Program_state_sampler/Sampler_results", f"init_{progname}_1.json"), "r") as f:
        init_state = json.load(f)
    return init_state


def get_config_cand(progname):
    with open(os.path.join(PARENT_PATH, "candidate_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config




with open(os.path.join(PARENT_PATH, "program-list.txt"), "r") as f:
    prognames = f.read().strip().split("\n")

for progname in prognames:
    config = get_config_cand(progname)
    prog_variables = config["Program_variables"]["Bools"]
    cand = config["Candidate"]["Expression"]
    iterations = config["Program specification"]["iterations"]




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
    

def get_ratio_new(tree_copy, mutation, leaf, distCurr, validCurr, split_count_map):
    max_split_count = 100
    if mutation == 'split':
        try:
            
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




def get_ratio(tree_copy, mutation, leaf, distCurr, validCurr):
    if mutation == 'split':
        try:
            '''If trying to split a leaf node that has only one datapoint, it will raise a ValueError.
            In this case, we would like to discard this move and try another one.'''
            tree_copy.split_node(leaf)
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
    #tree.save_tree(f'{PATH}/ex11_after_second_phase')
    distCurr, validCurr = error_bounds['dist_error'], error_bounds['valid_error']
    set_of_mutations = ['split', 'prune']
    tree_sequence = {}
    #while((distCurr > distUser or validCurr > validUser) and time.time() - start < 100):
    while(time.time() - start < 200):
        #print('{error_bounds=}', tree.get_error_bounds())
        leaf_ids = tree.get_all_leaf_nodes()
        #topk = round(len(clf.get_all_leaf_nodes())/2) + 1 
        moves = []
        for leaf in leaf_ids:
            for mutation in set_of_mutations:
                tree_copy = copy.deepcopy(tree)
                move = get_ratio(tree_copy, mutation, leaf, distCurr, validCurr)
                if move:
                    moves.append(move)
        #print("All moves:", [(move[0], move[3], move[4]) for move in moves])
        if not moves:
            print("No valid moves found. Exiting mutation loop.")
            break  # Exit the loop if no moves are available
        moves = Normalize(moves)
        moves.sort(key=lambda x: x[0], reverse=True)
        #print(moves)
        _, tree, error_bounds, leaf_id, mutation = random.choices(moves, weights=[move[0] for move in moves])[0]
        #print("leaf_id and mutation", leaf_id, mutation)
        #tree.save_tree(f'{PATH}/intermediate/ex10_{leaf_id}_{mutation}')
        tree_sequence[tree] = error_bounds
        distCurr, validCurr = error_bounds['dist_error'], error_bounds['valid_error']
        if distCurr <= distUser  and validCurr <= validUser:
            print("Successfully reached the desired user error bounds.")
            return tree, tree_sequence
    '''If timeout occurs, we return the tree with the best error bounds so far by sorting the dictionary tree_sequence and extracting the first element.'''
    print("Timeout occurred. Returning the best tree so far.")
    #print(tree.print_leaf_datapoints())
    if not tree_sequence:
        print("No valid tree sequence. Returning original tree.")
        #print(tree.print_leaf_datapoints())
        return tree, {}
    # best_tree = sorted(tree_sequence.items(), key=lambda x: round(np.sqrt(x[1]['dist_error'] + x[1]['valid_error'])))[0][0]
    best_tree = sorted(tree_sequence.items(), key=lambda x: round(np.sqrt(x[1]['dist_error'] + x[1]['valid_error']), 2))[0][0]
    #print("Best Tree Error Bounds:", best_tree.get_error_bounds())
    return best_tree, tree_sequence


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



def DNF_to_CNF_old(dnf_string):
    def parse_clause(clause):
        """Convert a conjunction clause into a set of literals."""
        return set(literal.strip() for literal in clause.split("&&"))

    def parse_dnf(dnf):
        """Parse a DNF string into a list of conjunction clauses."""
        clauses = dnf.split("||")
        return [parse_clause(clause.strip(" ")) for clause in clauses]

    def find_common_literals(clauses):
        """Find literals common across all clauses."""
        common = set.intersection(*clauses)
        return common

    def remove_common_literals(clauses, common):
        """Remove common literals from all clauses."""
        return [clause - common for clause in clauses]

    def simplify_clauses(clauses):
        """Simplify by removing redundant or subsumed clauses."""
        simplified = []
        for clause in clauses:
            if not any(clause < other for other in clauses):
                simplified.append(clause)
        return simplified

    def cnf_to_string(cnf_clauses):
        """Convert CNF clauses to string representation."""
        return " && ".join(
            "(" + " || ".join(literal for literal in clause) + ")" for clause in cnf_clauses
        )

    # Parse the DNF into clauses
    dnf_clauses = parse_dnf(dnf_string)

    # Find common literals across all clauses
    common_literals = find_common_literals(dnf_clauses)

    # Remove common literals from clauses
    reduced_clauses = remove_common_literals(dnf_clauses, common_literals)

    # Simplify the remaining clauses
    simplified_clauses = simplify_clauses(reduced_clauses)

    # Reconstruct the CNF: Add back the common literals to each simplified clause
    cnf_clauses = [clause | common_literals for clause in simplified_clauses]

    # Convert to CNF string format
    return cnf_to_string(cnf_clauses)


from z3 import *

def DNF_to_CNF(dnf_str):
    """
    Converts a DNF formula string to an equivalent CNF formula string using the Z3 solver.

    Args:
        dnf_str (str): A string representing the DNF formula, e.g., "!x1 && x2 || x3 && !x4".

    Returns:
        str: A string representing the equivalent CNF formula.
    """
    # Parse the DNF string into a Z3 expression
    def parse_literal(literal):
        literal = literal.strip()
        if literal.startswith("!"):
            return Not(Bool(literal[1:]))
        return Bool(literal)

    def parse_clause(clause):
        return And([parse_literal(lit) for lit in clause.split("&&")])

    dnf_clauses = dnf_str.split("||")
    dnf_expr = Or([parse_clause(clause.strip()) for clause in dnf_clauses])

    # Simplify and extract the CNF expression
    cnf_expr = Tactic('tseitin-cnf')(dnf_expr).as_expr()

    # Convert CNF Z3 expression back to a string
    def format_z3_expr(expr):
        if is_not(expr):
            return f"!{format_z3_expr(expr.arg(0))}"
        elif is_or(expr):
            return " || ".join(format_z3_expr(arg) for arg in expr.children())
        elif is_and(expr):
            return " && ".join(f"({format_z3_expr(arg)})" for arg in expr.children())
        else:
            return str(expr)

    return format_z3_expr(cnf_expr)






# initial_dist_data = [(0, 1, 1), (2, 1, 1), (5, 1, 1), (7, 1, 1), (8, 1, 1), (10, 1, 1), (13, 1, 1), (15, 1, 1), (16, 1, 1), (21, 1, 1), (23, 1, 1), (24, 1, 1), (26, 1, 1), (29, 1, 1), (31, 1, 1), (64, 1, 1), (69, 1, 1), (71, 1, 1), (72, 1, 1), (74, 1, 1), (80, 1, 1), (87, 1, 1), (90, 1, 1), (95, 1, 1), (160, 1, 1), (162, 1, 1), (165, 1, 1), (167, 1, 1), (168, 1, 1), (170, 1, 1), (173, 1, 1), (175, 1, 1), (176, 1, 1), (181, 1, 1), (183, 1, 1), (184, 1, 1), (186, 1, 1), (189, 1, 1), (191, 1, 1), (226, 1, 1), (229, 1, 1), (231, 1, 1), (240, 1, 1), (245, 1, 1), (247, 1, 1), (250, 1, 1), (253, 1, 1), (255, 1, 1)]
# initial_valid_data = [(1, 1, 0), (3, 1, 0), (6, 1, 0), (9, 1, 0), (12, 1, 0), (17, 1, 0), (19, 1, 0), (20, 1, 0), (22, 1, 0), (28, 1, 0), (30, 1, 0), (32, 1, 0), (34, 1, 0), (35, 1, 0), (39, 1, 0), (43, 1, 0), (45, 1, 0), (46, 1, 0), (49, 1, 0), (52, 1, 0), (55, 1, 0), (60, 1, 0), (63, 1, 0), (65, 1, 0), (68, 1, 0), (73, 1, 0), (76, 1, 0), (78, 1, 0), (81, 1, 0), (89, 1, 0), (91, 1, 0), (92, 1, 0), (97, 1, 0), (100, 1, 0), (101, 1, 0), (104, 1, 0), (105, 1, 0), (107, 1, 0), (108, 1, 0), (112, 1, 0), (122, 1, 0), (123, 1, 0), (125, 1, 0), (129, 1, 0), (130, 1, 0), (131, 1, 0), (134, 1, 0), (141, 1, 0), (146, 1, 0), (147, 1, 0), (151, 1, 0), (159, 1, 0), (161, 1, 0), (163, 1, 0), (164, 1, 0), (169, 1, 0), (172, 1, 0), (174, 1, 0), (192, 1, 0), (194, 1, 0), (195, 1, 0), (197, 1, 0), (201, 1, 0), (202, 1, 0), (203, 1, 0), (206, 1, 0), (207, 1, 0), (208, 1, 0), (211, 1, 0), (212, 1, 0), (216, 1, 0), (220, 1, 0), (222, 1, 0), (225, 1, 0), (230, 1, 0), (233, 1, 0), (236, 1, 0), (238, 1, 0), (241, 1, 0), (243, 1, 0), (251, 1, 0), (252, 1, 0)]
# second_phase_dist_data = [(113, 1, 1), (245, 1, 1), (215, 1, 1), (247, 1, 1), (63, 1, 1), (37, 1, 1), (49, 1, 1), (133, 1, 1), (33, 1, 1), (39, 1, 1), (97, 1, 1), (151, 1, 1), (55, 1, 1), (101, 1, 1)]
# second_phase_valid_data = [(6, 1, 0), (11, 1, 0), (14, 1, 0), (22, 1, 0), (27, 1, 0), (30, 1, 0), (193, 1, 0), (194, 1, 0), (195, 1, 0), (68, 1, 0), (196, 1, 0), (198, 1, 0), (199, 1, 0), (200, 1, 0), (73, 1, 0), (202, 1, 0), (203, 1, 0), (204, 1, 0), (76, 1, 0), (78, 1, 0), (205, 1, 0), (208, 1, 0), (75, 1, 0), (201, 1, 0), (210, 1, 0), (206, 1, 0), (216, 1, 0), (218, 1, 0), (221, 1, 0), (223, 1, 0), (197, 1, 0), (225, 1, 0), (70, 1, 0), (227, 1, 0), (228, 1, 0), (230, 1, 0), (233, 1, 0), (235, 1, 0), (236, 1, 0), (238, 1, 0)]



'''
initial_dist_data = [(0, 1, 1), (1, 1, 1), (6, 1, 1), (7, 1, 1), (24, 1, 1), (25, 1, 1), (30, 1, 1), (31, 1, 1)]
initial_valid_data = [(5, 1, 0), (17, 1, 0), (19, 1, 0), (20, 1, 0), (21, 1, 0), (23, 1, 0), (26, 1, 0), (28, 1, 0)]
second_phase_dist_data = [(16, 1, 1), (28, 1, 1), (26, 1, 1), (18, 1, 1)]
second_phase_valid_data = [(2, 1, 0), (3, 1, 0), (4, 1, 0), (8, 1, 0), (9, 1, 0), (10, 1, 0), (11, 1, 0), (14, 1, 0), (15, 1, 0), (27, 1, 0), (29, 1, 0)]
third_phase_dist_data = [(12,1,1)]
third_phase_valid_data = [(17, 1, 0), (19, 1, 0), (20, 1, 0), (21, 1, 0), (22, 1, 0), (23, 1, 0), (27, 1, 0), (29, 1, 0)]
fourth_phase_dist_data = []
fourth_phase_valid_data = [(2, 1, 0), (4, 1, 0), (27, 1, 0), (29, 1, 0)]
fifth_phase_dist_data = []
fifth_phase_valid_data = [(2, 1, 0), (4, 1, 0), (27, 1, 0), (29, 1, 0)]
'''


#EX10 data
#initial_dist_data = [(521, 1, 1), (3456, 1, 1), (3583, 1, 1), (3894, 1, 1), (4253, 1, 1), (5392, 1, 1), (6311, 1, 1), (8833, 1, 1), (14168, 1, 1), (14183, 1, 1), (14251, 1, 1), (14480, 1, 1), (15030, 1, 1), (15681, 1, 1), (16187, 1, 1), (19077, 1, 1), (19869, 1, 1), (22549, 1, 1), (22669, 1, 1), (22728, 1, 1), (25254, 1, 1), (27201, 1, 1), (28907, 1, 1), (29377, 1, 1), (30830, 1, 1), (32694, 1, 1), (33304, 1, 1), (34248, 1, 1), (36287, 1, 1), (40954, 1, 1), (42252, 1, 1), (42411, 1, 1), (42798, 1, 1), (47782, 1, 1), (50969, 1, 1), (51047, 1, 1), (51107, 1, 1), (51390, 1, 1), (51442, 1, 1), (54597, 1, 1), (56657, 1, 1), (56803, 1, 1), (57436, 1, 1), (57531, 1, 1), (58640, 1, 1), (59939, 1, 1), (60885, 1, 1), (63475, 1, 1), (64065, 1, 1), (65673, 1, 1), (67063, 1, 1), (67562, 1, 1), (67827, 1, 1), (68096, 1, 1), (68138, 1, 1), (69777, 1, 1), (71591, 1, 1), (73544, 1, 1), (75528, 1, 1), (75584, 1, 1), (75627, 1, 1), (77084, 1, 1), (78488, 1, 1), (81284, 1, 1), (83929, 1, 1), (85799, 1, 1), (87402, 1, 1), (87952, 1, 1), (88807, 1, 1), (90265, 1, 1), (92350, 1, 1), (94916, 1, 1), (96191, 1, 1), (97647, 1, 1), (98192, 1, 1), (99054, 1, 1), (100275, 1, 1), (104294, 1, 1), (105191, 1, 1), (107126, 1, 1), (107254, 1, 1), (107882, 1, 1), (108757, 1, 1), (112005, 1, 1), (112562, 1, 1), (112567, 1, 1), (112755, 1, 1), (115303, 1, 1), (116911, 1, 1), (117418, 1, 1), (119550, 1, 1), (122279, 1, 1), (122343, 1, 1), (124808, 1, 1), (125633, 1, 1), (126462, 1, 1), (129643, 1, 1), (130403, 1, 1), (130934, 1, 1)]
#initial_valid_data = [(359, 1, 0), (1126, 1, 0), (6998, 1, 0), (8489, 1, 0), (9868, 1, 0), (13676, 1, 0), (14274, 1, 0), (14756, 1, 0), (15159, 1, 0), (16070, 1, 0), (19714, 1, 0), (23616, 1, 0), (24289, 1, 0), (24735, 1, 0), (26347, 1, 0), (29092, 1, 0), (31691, 1, 0), (32928, 1, 0), (33457, 1, 0), (33973, 1, 0), (36767, 1, 0), (37564, 1, 0), (38511, 1, 0), (38867, 1, 0), (39854, 1, 0), (40675, 1, 0), (41932, 1, 0), (43876, 1, 0), (44670, 1, 0), (46490, 1, 0), (49840, 1, 0), (50570, 1, 0), (54510, 1, 0), (56875, 1, 0), (59691, 1, 0), (60459, 1, 0), (61327, 1, 0), (61766, 1, 0), (62112, 1, 0), (62311, 1, 0), (63528, 1, 0), (64410, 1, 0), (65198, 1, 0), (65914, 1, 0), (66093, 1, 0), (67867, 1, 0), (68370, 1, 0), (68477, 1, 0), (69934, 1, 0), (71754, 1, 0), (72885, 1, 0), (73893, 1, 0), (74103, 1, 0), (75407, 1, 0), (76104, 1, 0), (77347, 1, 0), (79038, 1, 0), (79209, 1, 0), (79784, 1, 0), (81329, 1, 0), (85364, 1, 0), (89504, 1, 0), (90306, 1, 0), (90489, 1, 0), (93776, 1, 0), (94196, 1, 0), (95094, 1, 0), (95802, 1, 0), (97431, 1, 0), (98344, 1, 0), (99369, 1, 0), (100756, 1, 0), (103882, 1, 0), (105110, 1, 0), (105304, 1, 0), (105305, 1, 0), (108829, 1, 0), (112239, 1, 0), (113507, 1, 0), (114695, 1, 0), (116502, 1, 0), (117403, 1, 0), (117772, 1, 0), (117867, 1, 0), (118390, 1, 0), (118871, 1, 0), (119662, 1, 0), (119956, 1, 0), (120154, 1, 0), (120774, 1, 0), (123023, 1, 0), (123690, 1, 0), (126583, 1, 0), (126973, 1, 0), (126994, 1, 0), (127828, 1, 0), (128135, 1, 0), (129454, 1, 0), (130783, 1, 0)]
#second_phase_dist_data = [(101134, 1, 1), (101894, 1, 1), (49422, 1, 1), (108878, 1, 1), (109638, 1, 1), (51582, 1, 1), (66366, 1, 1), (66430, 1, 1), (68366, 1, 1), (69126, 1, 1), (126470, 1, 1), (125710, 1, 1), (126590, 1, 1), (68478, 1, 1), (109817, 1, 1), (98622, 1, 1), (69166, 1, 1), (117566, 1, 1), (118318, 1, 1), (100622, 1, 1), (101182, 1, 1), (101246, 1, 1), (102137, 1, 1), (117630, 1, 1), (66318, 1, 1), (99134, 1, 1), (65854, 1, 1), (125310, 1, 1), (67966, 1, 1), (125198, 1, 1), (67854, 1, 1), (68614, 1, 1), (101382, 1, 1), (109574, 1, 1), (117766, 1, 1), (118521, 1, 1), (115582, 1, 1), (117118, 1, 1), (118009, 1, 1), (98574, 1, 1), (69369, 1, 1), (52862, 1, 1), (52094, 1, 1), (98686, 1, 1), (49534, 1, 1), (115470, 1, 1), (126022, 1, 1), (100734, 1, 1), (101625, 1, 1), (50046, 1, 1), (99198, 1, 1), (52985, 1, 1), (110329, 1, 1), (125246, 1, 1), (67902, 1, 1), (68414, 1, 1), (65918, 1, 1), (114958, 1, 1), (110086, 1, 1), (108814, 1, 1), (101502, 1, 1), (117054, 1, 1), (76806, 1, 1), (125958, 1, 1), (125822, 1, 1), (69246, 1, 1), (101422, 1, 1), (100670, 1, 1), (99086, 1, 1), (68734, 1, 1), (52350, 1, 1), (51518, 1, 1), (115518, 1, 1), (49934, 1, 1), (115070, 1, 1), (49982, 1, 1), (126713, 1, 1), (110150, 1, 1), (109390, 1, 1), (109438, 1, 1), (68654, 1, 1), (117806, 1, 1), (51470, 1, 1), (52030, 1, 1), (109374, 1, 1), (49470, 1, 1), (117518, 1, 1), (118278, 1, 1), (77382, 1, 1), (126534, 1, 1), (126078, 1, 1), (76870, 1, 1), (125262, 1, 1), (68857, 1, 1), (118398, 1, 1), (76670, 1, 1), (117886, 1, 1), (115006, 1, 1), (102014, 1, 1), (108926, 1, 1), (109694, 1, 1), (125758, 1, 1), (65806, 1, 1), (108862, 1, 1), (126510, 1, 1)]
#second_phase_valid_data = [(45063, 1, 0), (45066, 1, 0), (26638, 1, 0), (47140, 1, 0), (12333, 1, 0), (30768, 1, 0), (26677, 1, 0), (49218, 1, 0), (53315, 1, 0), (16455, 1, 0), (43082, 1, 0), (26699, 1, 0), (61514, 1, 0), (12366, 1, 0), (20562, 1, 0), (30803, 1, 0), (2135, 1, 0), (24666, 1, 0), (53338, 1, 0), (39006, 1, 0), (51294, 1, 0), (10337, 1, 0), (2148, 1, 0), (4208, 1, 0), (39029, 1, 0), (18556, 1, 0), (6269, 1, 0), (30863, 1, 0), (37011, 1, 0), (22682, 1, 0), (26782, 1, 0), (2216, 1, 0), (14505, 1, 0), (14512, 1, 0), (14520, 1, 0), (4291, 1, 0), (24783, 1, 0), (30939, 1, 0), (47328, 1, 0), (47329, 1, 0), (22753, 1, 0), (55524, 1, 0), (10473, 1, 0), (55536, 1, 0), (20721, 1, 0), (61825, 1, 0), (55687, 1, 0), (55688, 1, 0), (8585, 1, 0), (43400, 1, 0), (45451, 1, 0), (51594, 1, 0), (18831, 1, 0), (20883, 1, 0), (10647, 1, 0), (51607, 1, 0), (22938, 1, 0), (412, 1, 0), (10671, 1, 0), (53680, 1, 0), (22961, 1, 0), (29109, 1, 0), (27064, 1, 0), (35257, 1, 0), (12733, 1, 0), (43456, 1, 0), (51649, 1, 0), (45506, 1, 0), (27076, 1, 0), (16842, 1, 0), (4554, 1, 0), (57805, 1, 0), (10701, 1, 0), (27087, 1, 0), (35284, 1, 0), (16852, 1, 0), (43479, 1, 0), (45527, 1, 0), (4568, 1, 0), (63962, 1, 0), (25047, 1, 0), (18909, 1, 0), (41439, 1, 0), (33256, 1, 0), (16875, 1, 0), (23020, 1, 0), (47597, 1, 0), (25070, 1, 0), (33262, 1, 0), (4595, 1, 0), (47611, 1, 0), (49660, 1, 0), (25085, 1, 0), (10754, 1, 0), (25094, 1, 0), (10759, 1, 0), (14859, 1, 0), (59919, 1, 0), (2579, 1, 0), (35358, 1, 0), (31265, 1, 0), (35365, 1, 0), (12848, 1, 0), (59953, 1, 0), (21044, 1, 0), (10819, 1, 0), (25171, 1, 0), (35414, 1, 0), (64086, 1, 0), (29271, 1, 0), (19039, 1, 0), (43620, 1, 0), (2664, 1, 0), (12908, 1, 0), (12909, 1, 0), (19052, 1, 0), (17012, 1, 0), (43641, 1, 0), (35453, 1, 0), (55939, 1, 0), (650, 1, 0), (51867, 1, 0), (37531, 1, 0), (58033, 1, 0), (39622, 1, 0), (29394, 1, 0), (10971, 1, 0), (4831, 1, 0), (56032, 1, 0), (35560, 1, 0), (25321, 1, 0), (62196, 1, 0), (33525, 1, 0), (31485, 1, 0), (50048, 1, 0), (29574, 1, 0), (64396, 1, 0), (21389, 1, 0), (64398, 1, 0), (35726, 1, 0), (19341, 1, 0), (33694, 1, 0), (50084, 1, 0), (31652, 1, 0), (27557, 1, 0), (29607, 1, 0), (19368, 1, 0), (35753, 1, 0), (29610, 1, 0), (35755, 1, 0), (37804, 1, 0), (43949, 1, 0), (7088, 1, 0), (15285, 1, 0), (13240, 1, 0), (41913, 1, 0), (50104, 1, 0), (13244, 1, 0), (37822, 1, 0), (62402, 1, 0), (48070, 1, 0), (35784, 1, 0), (43976, 1, 0), (21453, 1, 0), (41940, 1, 0), (37846, 1, 0), (33755, 1, 0), (25567, 1, 0), (3043, 1, 0), (3046, 1, 0), (48103, 1, 0), (35814, 1, 0), (5101, 1, 0), (7150, 1, 0), (35822, 1, 0), (29679, 1, 0), (13298, 1, 0), (15349, 1, 0), (23541, 1, 0), (56315, 1, 0), (21500, 1, 0), (56317, 1, 0), (27784, 1, 0), (44168, 1, 0), (40074, 1, 0), (5260, 1, 0), (62604, 1, 0), (58508, 1, 0), (23700, 1, 0), (40086, 1, 0), (25760, 1, 0), (11426, 1, 0), (25766, 1, 0), (5288, 1, 0), (11434, 1, 0), (15534, 1, 0), (27822, 1, 0), (58544, 1, 0), (13490, 1, 0), (25782, 1, 0), (40126, 1, 0), (48326, 1, 0), (25798, 1, 0), (56530, 1, 0), (62680, 1, 0), (23774, 1, 0), (48358, 1, 0), (64746, 1, 0), (29934, 1, 0), (13574, 1, 0), (56591, 1, 0), (64790, 1, 0), (7450, 1, 0), (21790, 1, 0), (58665, 1, 0), (44329, 1, 0), (44336, 1, 0), (30005, 1, 0), (52537, 1, 0), (32057, 1, 0), (5442, 1, 0), (27974, 1, 0), (27979, 1, 0), (25934, 1, 0), (46419, 1, 0), (3414, 1, 0), (15720, 1, 0), (9588, 1, 0), (46457, 1, 0), (23932, 1, 0), (9602, 1, 0), (62859, 1, 0), (5526, 1, 0), (32161, 1, 0), (13740, 1, 0), (64945, 1, 0), (60853, 1, 0), (5560, 1, 0), (19901, 1, 0), (46550, 1, 0), (48624, 1, 0), (58865, 1, 0), (19956, 1, 0), (34297, 1, 0), (65152, 1, 0), (38532, 1, 0), (24202, 1, 0), (18058, 1, 0), (5772, 1, 0), (48784, 1, 0), (63130, 1, 0), (48796, 1, 0), (5788, 1, 0), (32414, 1, 0), (26276, 1, 0), (50856, 1, 0), (16040, 1, 0), (22188, 1, 0), (1708, 1, 0), (57012, 1, 0), (65206, 1, 0), (5822, 1, 0), (44738, 1, 0), (52938, 1, 0), (1742, 1, 0), (57046, 1, 0), (5848, 1, 0), (12002, 1, 0), (20196, 1, 0), (38636, 1, 0), (14083, 1, 0), (24323, 1, 0), (18183, 1, 0), (28427, 1, 0), (26387, 1, 0), (30494, 1, 0), (42788, 1, 0), (22309, 1, 0), (57128, 1, 0), (57129, 1, 0), (28460, 1, 0), (55085, 1, 0), (57140, 1, 0), (24373, 1, 0), (1848, 1, 0), (42818, 1, 0), (24387, 1, 0), (57155, 1, 0), (65350, 1, 0), (32583, 1, 0), (28486, 1, 0), (48971, 1, 0), (32595, 1, 0), (10079, 1, 0), (1888, 1, 0), (8032, 1, 0), (3937, 1, 0), (1893, 1, 0), (30568, 1, 0), (12137, 1, 0), (38761, 1, 0), (16244, 1, 0), (14197, 1, 0), (65401, 1, 0), (32642, 1, 0), (32654, 1, 0), (44959, 1, 0), (61345, 1, 0), (26537, 1, 0), (26548, 1, 0), (38837, 1, 0), (47037, 1, 0), (8135, 1, 0), (42958, 1, 0), (63454, 1, 0), (51168, 1, 0), (55268, 1, 0), (65509, 1, 0), (36840, 1, 0), (49141, 1, 0), (20476, 1, 0), (36861, 1, 0)]
#third_phase_dist_data = 

#Ex12 data goes here
#initial_dist_data = [(10, 1, 1), (24, 1, 1), (33, 1, 1), (43, 1, 1), (65, 1, 1), (83, 1, 1), (96, 1, 1), (107, 1, 1), (277, 1, 1), (309, 1, 1), (316, 1, 1), (334, 1, 1), (547, 1, 1), (576, 1, 1), (585, 1, 1), (587, 1, 1), (593, 1, 1), (595, 1, 1), (618, 1, 1), (780, 1, 1), (789, 1, 1), (822, 1, 1), (838, 1, 1), (885, 1, 1), (1075, 1, 1), (1082, 1, 1), (1104, 1, 1), (1123, 1, 1), (1139, 1, 1), (1144, 1, 1), (1145, 1, 1), (1300, 1, 1), (1319, 1, 1), (1326, 1, 1), (1349, 1, 1), (1365, 1, 1), (1563, 1, 1), (1576, 1, 1), (1584, 1, 1), (1595, 1, 1), (1619, 1, 1), (1627, 1, 1), (1634, 1, 1), (1799, 1, 1), (1854, 1, 1), (1876, 1, 1), (1884, 1, 1), (1894, 1, 1), (1901, 1, 1), (2176, 1, 1), (2186, 1, 1), (2192, 1, 1), (2195, 1, 1), (2201, 1, 1), (2209, 1, 1), (2233, 1, 1), (2249, 1, 1), (2251, 1, 1), (2275, 1, 1), (2462, 1, 1), (2503, 1, 1), (2517, 1, 1), (2697, 1, 1), (2705, 1, 1), (2728, 1, 1), (2730, 1, 1), (2792, 1, 1), (2988, 1, 1), (3014, 1, 1), (3046, 1, 1), (3281, 1, 1), (3282, 1, 1), (3299, 1, 1), (3321, 1, 1), (3468, 1, 1), (3493, 1, 1), (3500, 1, 1), (3502, 1, 1), (3516, 1, 1), (3564, 1, 1), (3712, 1, 1), (3713, 1, 1), (3737, 1, 1), (3761, 1, 1), (3786, 1, 1), (3817, 1, 1), (3819, 1, 1), (3835, 1, 1), (3991, 1, 1), (4014, 1, 1), (4045, 1, 1), (4053, 1, 1), (4087, 1, 1), (4094, 1, 1)]
#initial_valid_data = [(95, 1, 0), (125, 1, 0), (154, 1, 0), (195, 1, 0), (232, 1, 0), (272, 1, 0), (393, 1, 0), (421, 1, 0), (451, 1, 0), (489, 1, 0), (549, 1, 0), (710, 1, 0), (742, 1, 0), (856, 1, 0), (910, 1, 0), (911, 1, 0), (932, 1, 0), (956, 1, 0), (968, 1, 0), (969, 1, 0), (1047, 1, 0), (1193, 1, 0), (1208, 1, 0), (1218, 1, 0), (1255, 1, 0), (1329, 1, 0), (1345, 1, 0), (1429, 1, 0), (1438, 1, 0), (1472, 1, 0), (1488, 1, 0), (1489, 1, 0), (1597, 1, 0), (1607, 1, 0), (1669, 1, 0), (1677, 1, 0), (1696, 1, 0), (1755, 1, 0), (1760, 1, 0), (1771, 1, 0), (1783, 1, 0), (1809, 1, 0), (1832, 1, 0), (1882, 1, 0), (1883, 1, 0), (1941, 1, 0), (1946, 1, 0), (1995, 1, 0), (2008, 1, 0), (2023, 1, 0), (2046, 1, 0), (2075, 1, 0), (2112, 1, 0), (2145, 1, 0), (2159, 1, 0), (2165, 1, 0), (2295, 1, 0), (2334, 1, 0), (2372, 1, 0), (2381, 1, 0), (2481, 1, 0), (2489, 1, 0), (2505, 1, 0), (2593, 1, 0), (2607, 1, 0), (2734, 1, 0), (2835, 1, 0), (2876, 1, 0), (2898, 1, 0), (2947, 1, 0), (2992, 1, 0), (2994, 1, 0), (3104, 1, 0), (3133, 1, 0), (3143, 1, 0), (3214, 1, 0), (3222, 1, 0), (3270, 1, 0), (3316, 1, 0), (3347, 1, 0), (3368, 1, 0), (3426, 1, 0), (3588, 1, 0), (3622, 1, 0), (3650, 1, 0), (3658, 1, 0), (3750, 1, 0), (3797, 1, 0), (3830, 1, 0), (3867, 1, 0), (3902, 1, 0), (3907, 1, 0), (3944, 1, 0), (4049, 1, 0)]
#second_phase_dist_data = [(3125, 1, 1), (3124, 1, 1), (3952, 1, 1), (1904, 1, 1), (2212, 1, 1), (2085, 1, 1), (2084, 1, 1), (3058, 1, 1), (2292, 1, 1), (880, 1, 1), (1010, 1, 1), (1268, 1, 1), (2898, 1, 1), (2213, 1, 1), (2899, 1, 1), (3440, 1, 1), (3316, 1, 1), (1018, 1, 1), (172, 1, 1), (173, 1, 1), (2417, 1, 1), (3253, 1, 1), (4083, 1, 1), (4082, 1, 1), (3441, 1, 1), (1019, 1, 1), (2416, 1, 1), (244, 1, 1), (3059, 1, 1), (889, 1, 1), (2034, 1, 1), (3252, 1, 1), (888, 1, 1)]
#second_phase_valid_data = [(1024, 1, 0), (509, 1, 0), (0, 1, 0), (1539, 1, 0), (2, 1, 0), (1032, 1, 0), (1033, 1, 0), (521, 1, 0), (511, 1, 0), (530, 1, 0), (26, 1, 0), (539, 1, 0), (32, 1, 0), (545, 1, 0), (1569, 1, 0), (1571, 1, 0), (35, 1, 0), (1568, 1, 0), (552, 1, 0), (41, 1, 0), (42, 1, 0), (43, 1, 0), (1067, 1, 0), (1066, 1, 0), (1584, 1, 0), (49, 1, 0), (50, 1, 0), (1072, 1, 0), (1585, 1, 0), (568, 1, 0), (1593, 1, 0), (1082, 1, 0), (1083, 1, 0), (1600, 1, 0), (1088, 1, 0), (1090, 1, 0), (578, 1, 0), (64, 1, 0), (1089, 1, 0), (1616, 1, 0), (1617, 1, 0), (1106, 1, 0), (1112, 1, 0), (1113, 1, 0), (91, 1, 0), (1632, 1, 0), (1633, 1, 0), (1635, 1, 0), (99, 1, 0), (1123, 1, 0), (1640, 1, 0), (105, 1, 0), (617, 1, 0), (1641, 1, 0), (1649, 1, 0), (1139, 1, 0), (632, 1, 0), (1657, 1, 0), (1658, 1, 0), (634, 1, 0), (121, 1, 0), (123, 1, 0), (122, 1, 0), (1147, 1, 0), (1152, 1, 0), (1153, 1, 0), (642, 1, 0), (1667, 1, 0), (1154, 1, 0), (1160, 1, 0), (1673, 1, 0), (1162, 1, 0), (1680, 1, 0), (145, 1, 0), (1681, 1, 0), (1171, 1, 0), (154, 1, 0), (1178, 1, 0), (1697, 1, 0), (1185, 1, 0), (675, 1, 0), (161, 1, 0), (169, 1, 0), (681, 1, 0), (683, 1, 0), (176, 1, 0), (689, 1, 0), (688, 1, 0), (1713, 1, 0), (691, 1, 0), (177, 1, 0), (184, 1, 0), (1721, 1, 0), (1209, 1, 0), (1723, 1, 0), (697, 1, 0), (1720, 1, 0), (1216, 1, 0), (704, 1, 0), (1729, 1, 0), (707, 1, 0), (712, 1, 0), (1225, 1, 0), (202, 1, 0), (1738, 1, 0), (1744, 1, 0), (1745, 1, 0), (1234, 1, 0), (721, 1, 0), (209, 1, 0), (728, 1, 0), (1243, 1, 0), (1248, 1, 0), (1250, 1, 0), (227, 1, 0), (1251, 1, 0), (232, 1, 0), (1257, 1, 0), (234, 1, 0), (235, 1, 0), (1771, 1, 0), (1777, 1, 0), (1778, 1, 0), (1784, 1, 0), (761, 1, 0), (260, 1, 0), (261, 1, 0), (2311, 1, 0), (3847, 1, 0), (268, 1, 0), (3855, 1, 0), (2831, 1, 0), (788, 1, 0), (1302, 1, 0), (278, 1, 0), (1308, 1, 0), (1821, 1, 0), (285, 1, 0), (797, 1, 0), (798, 1, 0), (2847, 1, 0), (1318, 1, 0), (300, 1, 0), (301, 1, 0), (1325, 1, 0), (1836, 1, 0), (812, 1, 0), (1844, 1, 0), (821, 1, 0), (1845, 1, 0), (3895, 1, 0), (316, 1, 0), (2879, 1, 0), (1861, 1, 0), (1862, 1, 0), (3399, 1, 0), (2375, 1, 0), (1869, 1, 0), (1870, 1, 0), (846, 1, 0), (852, 1, 0), (342, 1, 0), (3927, 1, 0), (3415, 1, 0), (2903, 1, 0), (1884, 1, 0), (1372, 1, 0), (1886, 1, 0), (2399, 1, 0), (3935, 1, 0), (861, 1, 0), (1381, 1, 0), (357, 1, 0), (2919, 1, 0), (364, 1, 0), (1389, 1, 0), (365, 1, 0), (3439, 1, 0), (2415, 1, 0), (372, 1, 0), (884, 1, 0), (3447, 1, 0), (1916, 1, 0), (893, 1, 0), (1406, 1, 0), (381, 1, 0), (894, 1, 0), (1927, 1, 0), (909, 1, 0), (397, 1, 0), (1940, 1, 0), (407, 1, 0), (919, 1, 0), (1436, 1, 0), (1438, 1, 0), (1951, 1, 0), (1444, 1, 0), (933, 1, 0), (940, 1, 0), (1964, 1, 0), (430, 1, 0), (1967, 1, 0), (1966, 1, 0), (948, 1, 0), (1462, 1, 0), (950, 1, 0), (446, 1, 0), (1988, 1, 0), (966, 1, 0), (973, 1, 0), (1487, 1, 0), (2012, 1, 0), (477, 1, 0), (1500, 1, 0), (484, 1, 0), (996, 1, 0), (486, 1, 0), (487, 1, 0), (1517, 1, 0), (2030, 1, 0), (495, 1, 0), (2036, 1, 0), (1013, 1, 0), (1020, 1, 0), (2045, 1, 0), (1535, 1, 0)]



DistEstimate_trigger_path = os.path.join(PARENT_PATH, "src/DistEstimate/", "DistEstimate_trigger.sh")
Validifier_trigger_path = os.path.join(PARENT_PATH, "src/Validifier/", "Validifier_trigger.sh")

#DistEstimate_trigger_path = os.path.join(PATH, "DistEstimate_trigger.sh")
#Validifier_trigger_path = os.path.join(PATH, "Validifier_trigger.sh")

start = time.time()
for progname in prognames:
    initial_states = get_init_state(progname)
    cand_info = get_config_cand(progname)


initial_phase_start = time.time()
initial_dist_list = initial_states["output_dict"]["Sampled positive initial states"]
initial_valid_list = initial_states["output_dict"]["Sampled negative initial states"]
initial_phase_dist_list = [int(element) for element in initial_dist_list]
initial_phase_valid_list = [int(element) for element in initial_valid_list]
initial_dist_data = [(state, 1, 1) for state in initial_phase_dist_list]
initial_valid_data = [(state, 1, 0) for state in initial_phase_valid_list]
print("Initial dist data:",initial_dist_data)
print("Initial valid data:",initial_valid_data)
#print(length := len(initial_dist_data), len(initial_valid_data), len(second_phase_dist_data), len(second_phase_valid_data))
#number_of_vars = 8
number_of_vars = 10
#number_of_vars = 5
#number_of_vars = 17
#number_of_vars = 12
df_dist = prepare_data(initial_dist_data, number_of_vars, DistEstimate = True)
df_valid = prepare_data(initial_valid_data, number_of_vars, DistEstimate = False)
df = pd.concat([df_dist, df_valid], ignore_index=True)
X = df[list(df.columns[:-3])].values
y = df['label'].values
clf = CustomDecisionTree(max_depth=number_of_vars)
clf.fit_initial(X, y, feature_names=df.columns[:-3])
clf.save_tree(f'{PATH}/ex9_before_prediction')
predictions = clf.predict(df_dist[list(df.columns[:-3])].values, expected_labels=df_dist['label'].values, weights=df_dist['weight'].values, member=df_dist['member'].values)
#print("Initial Bounds(after prediction):", clf.get_error_bounds())
clf.save_tree(f'{PATH}/ex9_after_prediction')
Init_tree_DNF = clf.tree_to_dnf()
#print("Initial phase DNF:", Init_tree_DNF)
print("Initial phase candidate DNF:", Init_tree_DNF)


print("Initial phase TreeLearner time: ", time.time() - initial_phase_start)
#print(cand_info)
with open(os.path.join(PARENT_PATH, "candidate_files", progname + ".json"), "r") as file:
    data = json.load(file)
    data["Candidate"]["Expression"] = Init_tree_DNF
with open(os.path.join(PARENT_PATH, "candidate_files", progname + ".json"), "w") as file:
    json.dump(data, file, indent=4)



#print("Running Validifier script located at:", Validifier_trigger_path)
#print("Running DistEstimate script located at:", DistEstimate_trigger_path)

initial_verifier_time = time.time()
try:
    # Run the first subprocess
    result_validifier = subprocess.run(
        ["bash", Validifier_trigger_path],
        capture_output=True,
        text=True,
        check=True  # Raise CalledProcessError if the subprocess fails
    )
    #print("Validifier Output:", result_validifier.stdout)
    

    # Run the second subprocess
    result_distestimate = subprocess.run(
        ["bash", DistEstimate_trigger_path],
        capture_output=True,
        text=True,
        check=True  # Raise CalledProcessError if the subprocess fails
    )
    #print("DistEstimate Output:", result_distestimate.stdout)
    

except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with return code {e.returncode}")
    print(f"Command: {e.cmd}")
    print(f"Output: {e.output}")
    print(f"Error: {e.stderr}")

except FileNotFoundError as e:
    print(f"Executable not found: {e}")
    print("Check that 'bash' is installed and paths are correct.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")



print("Initial Verifier time :", time.time() - initial_verifier_time)

valid_results_directory = os.path.join(PARENT_PATH, "src/Validifier/Validifier_results")
if not os.path.exists(valid_results_directory):
    os.makedirs(valid_results_directory)
existing_valid_files = os.listdir(valid_results_directory)
file_number = 1
while f"exp_{progname}_{file_number}.json" in existing_valid_files:
    file_number += 1
if file_number > 1:
    file_number = file_number - 1
#print(file_number)
validifier_path = os.path.join(PARENT_PATH, "src/Validifier/Validifier_results", f"exp_{progname}_{file_number}.json")
#print(f"Checking for validifier results at: {validifier_path}")
if not os.path.exists(validifier_path):
    print(f"Warning: No validifier results found at {validifier_path}")
    validifier_value = 0
    validifier_data = {"output_dict": {"counterexamples": {}, "Validifier_value": 0}}
else:
    with open(validifier_path, "r") as file:
        validifier_data = json.load(file)
        validifier_value = validifier_data["output_dict"]["Validifier_value"]
    




dist_results_directory = os.path.join(PARENT_PATH, "src/DistEstimate/DistEstimate_results")
if not os.path.exists(dist_results_directory):
    os.makedirs(dist_results_directory)
existing_dist_files = os.listdir(dist_results_directory)
file_number = 1
while f"exp_{progname}_{file_number}.json" in existing_dist_files:
    file_number += 1

if file_number > 1:
    file_number = file_number - 1
#print(file_number)

with open(os.path.join(PARENT_PATH, "src/DistEstimate/DistEstimate_results", f"exp_{progname}_{file_number}.json"), "r") as file:
    distestimate_data = json.load(file)
    #next_dist_list = distestimate_data["output_dict"]["counterexamples"]
    distestimate_value = distestimate_data["output_dict"]["DistEstimate_value"]



print("Initial phase time :", time.time() - initial_phase_start)
intermediate_tree = clf
while (distestimate_value != 0 or validifier_value != 0):

    print("DistEstimate value:",distestimate_value)
    print("Validifier value:",validifier_value)


    next_phase_start = time.time()
    next_dist_dict = distestimate_data["output_dict"]["counterexamples"]
    next_valid_dict = validifier_data["output_dict"]["counterexamples"]
    next_phase_dist_list = [int(element) for element in next_dist_dict.keys()]
    next_phase_valid_list = list(set(int(element) for element in next_valid_dict.keys()) - set(initial_phase_dist_list))
    next_phase_dist_data = [(int(element),1,1) for element in next_phase_dist_list]
    next_phase_valid_data = [(int(element),1,0) for element in next_phase_valid_list]
    print("next_phase_valid_data",next_phase_valid_data)
    print("next_phase_dist_data",next_phase_dist_data)
    df_dist_next = prepare_data(next_phase_dist_data, number_of_vars, DistEstimate = True)
    df_valid_next = prepare_data(next_phase_valid_data, number_of_vars, DistEstimate = False)
    df_next = pd.concat([df_dist_next, df_valid_next], ignore_index=True)
    intermediate_tree.predict(df_next[list(df.columns[:-3])].values, expected_labels=df_next['label'].values, weights=df_next['weight'].values, member=df_next['member'].values)
    #clf.print_leaf_datapoints()
    #print("Next Phase Bounds:", intermediate_tree.get_error_bounds())
    #final_tree.save_tree(f'{PATH}/ex10_after_next_phase')
    #print("DNF after prediction:", clf.tree_to_dnf())


    for iteration in range(5):
        intermediate_tree_copy = copy.deepcopy(intermediate_tree)
        print(f'{iteration=}')
        random.seed(iteration + 10)
        print("------NEXT PHASE------")
        final_tree, tree_sequence = MuteTree(intermediate_tree_copy, df_next, 0, 0)
        final_tree.save_tree(output_file=f'{PATH}/final_trees/ex9_second_{iteration}')
        Next_phase_DNF = final_tree.tree_to_dnf()
        print("Next_phase_DNF:", Next_phase_DNF)
        

    print("Next phase TreeLearner time :", time.time() - next_phase_start)
    with open(os.path.join(PARENT_PATH, "candidate_files", progname + ".json"), "r") as file:
        data = json.load(file)
        data["Candidate"]["Expression"] = Next_phase_DNF
    with open(os.path.join(PARENT_PATH, "candidate_files", progname + ".json"), "w") as file:
        json.dump(data, file, indent=4)


    next_phase_verifier_time = time.time()
    try:
        # Run the first subprocess
        result_validifier = subprocess.run(
            ["bash", Validifier_trigger_path],
            capture_output=True,
            text=True,
            check=True  # Raise CalledProcessError if the subprocess fails
        )
        #print("Validifier Output:", result_validifier.stdout)
        
    
    # Run the second subprocess
        result_distestimate = subprocess.run(
            ["bash", DistEstimate_trigger_path],
            capture_output=True,
            text=True,
            check=True  # Raise CalledProcessError if the subprocess fails
        )
        #print("DistEstimate Output:", result_distestimate.stdout)
    
    
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with return code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")

    except FileNotFoundError as e:
        print(f"Executable not found: {e}")
        print("Check that 'bash' is installed and paths are correct.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Next phase Verifier time: ", time.time() - next_phase_verifier_time)


    valid_results_directory = os.path.join(PARENT_PATH, "src/Validifier/Validifier_results")
    if not os.path.exists(valid_results_directory):
        os.makedirs(valid_results_directory)
    existing_valid_files = os.listdir(valid_results_directory)
    file_number = 1
    while f"exp_{progname}_{file_number}.json" in existing_valid_files:
        file_number += 1
    file_number = file_number - 1


    validifier_path = os.path.join(PARENT_PATH, "src/Validifier/Validifier_results", f"exp_{progname}_{file_number}.json")
    #print(f"Checking for validifier results at: {validifier_path}")
    if not os.path.exists(validifier_path):
        print(f"Warning: No validifier results found at {validifier_path}")
    else:
        with open(validifier_path, "r") as file:
            validifier_data = json.load(file)
            validifier_value = validifier_data["output_dict"]["Validifier_value"]
        

    dist_results_directory = os.path.join(PARENT_PATH, "src/DistEstimate/DistEstimate_results")
    if not os.path.exists(dist_results_directory):
        os.makedirs(dist_results_directory)
    existing_dist_files = os.listdir(dist_results_directory)
    file_number = 1
    while f"exp_{progname}_{file_number}.json" in existing_dist_files:
        file_number += 1
    file_number = file_number - 1
    with open(os.path.join(PARENT_PATH, "src/DistEstimate/DistEstimate_results", f"exp_{progname}_{file_number}.json"), "r") as file:
        distestimate_data = json.load(file)
        #second_dist_list = distestimate_data["output_dict"]["counterexamples"]
        distestimate_value = distestimate_data["output_dict"]["DistEstimate_value"]

    intermediate_tree = final_tree


    print("Next phase total time: ", time.time() - next_phase_start)



print("Final DistEstimate value: ",distestimate_value)
print("Final Validifier value: ",validifier_value)
print("Final candidate learnt: ",Next_phase_DNF)
print("Total time taken: ",time.time() - start)






'''
df_dist_third = prepare_data(third_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_third = prepare_data(third_phase_valid_data, number_of_vars, DistEstimate = False)
df_third = pd.concat([df_dist_third, df_valid_third], ignore_index=True)

df_dist_fourth = prepare_data(fourth_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_fourth = prepare_data(fourth_phase_valid_data, number_of_vars, DistEstimate = False)
df_fourth = pd.concat([df_dist_fourth, df_valid_fourth], ignore_index=True)


df_dist_fifth = prepare_data(fifth_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_fifth = prepare_data(fifth_phase_valid_data, number_of_vars, DistEstimate = False)
df_fifth = pd.concat([df_dist_fifth, df_valid_fifth], ignore_index=True)



for iteration in range(1):
    clf_copy = copy.deepcopy(clf)
    print(f'{iteration=}')
    random.seed(iteration + 10)
    print("------SECOND PHASE------")
    final_tree, tree_sequence = MuteTree(clf_copy, df_second, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex10_second_{iteration}')
    print("Second phase DNF:", final_tree.tree_to_dnf())
    
    
    print("------THIRD PHASE------")
    predictions = final_tree.predict(df_third[list(df.columns[:-3])].values, expected_labels=df_third['label'].values, weights=df_third['weight'].values, member=df_third['member'].values)
    print("Error bounds AFTER PREDICTING ON THIRD PHASE DATA: ",final_tree.get_error_bounds())
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex10_third_{iteration}_beginning')

    final_tree, tree_sequence = MuteTree(final_tree, df_third, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex10_third_{iteration}')
    print("Third phase DNF:", final_tree.tree_to_dnf())

    
    print("------FOURTH PHASE------")
    predictions = final_tree.predict(df_fourth[list(df.columns[:-3])].values, expected_labels=df_fourth['label'].values, weights=df_fourth['weight'].values, member=df_fourth['member'].values)
    print("Error bounds AFTER PREDICTING ON FOURTH PHASE DATA: ",final_tree.get_error_bounds())
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex10_fourth_{iteration}_beginning')

    final_tree, tree_sequence = MuteTree(final_tree, df_fourth, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex10_fourth_{iteration}')
    print("Fourth phase DNF:", final_tree.tree_to_dnf())


    print("------FIFTH PHASE------")
    predictions = final_tree.predict(df_fifth[list(df.columns[:-3])].values, expected_labels=df_fifth['label'].values, weights=df_fifth['weight'].values, member=df_fifth['member'].values)
    print("Error bounds AFTER PREDICTING ON FIFTH PHASE DATA: ",final_tree.get_error_bounds())
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex10_fifth_{iteration}_beginning')

    final_tree, tree_sequence = MuteTree(final_tree, df_fifth, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex10_fifth_{iteration}')
    print("Fifth phase DNF:", final_tree.tree_to_dnf())
    final_tree.print_leaf_datapoints()
    
    

'''






'''Example 2
initial_data = {'a': [0,0,0,1,1,1,1,1,0,0,0],
    'b': [0,0,0,1,1,1,0,0,1,1,1],
    'c': [0,0,1,0,0,1,1,0,0,1,1],
    'd': [1,1,0,1,1,0,1,0,1,0,1],
    'e': [0,1,0,0,1,1,0,1,0,0,1],
    'label': [1,1,1,1,1,1,0,0,0,0,0],
    'weight': [1,1,1,1,1,1,1,1,1,1,1],
    'member': [1,1,1,1,1,1,0,0,0,0,0]}

data2 = {'a': [0,0,0,0,0,0,0,0,1,1,1,1,1],
    'b': [0,0,0,0,1,1,1,1,0,0,0,1,1],
    'c': [0,0,1,1,0,0,0,1,0,0,1,0,0],
    'd': [0,0,1,1,0,0,1,1,1,1,0,0,0],
    'e': [0,1,0,1,0,1,1,0,0,1,0,0,1],
    'label': [1,1,1,1,0,0,0,0,0,0,0,1,1],
    'weight': [1,1,1,1,1,1,1,1,1,1,1,1,1],
    'member': [1,1,1,1,0,0,0,0,0,0,0,1,1]}

data3 = {'a': [0,0,1,1,1,1,1,1],
    'b': [0,1,0,0,0,1,1,1],
    'c': [1,1,0,1,1,1,1,1],
    'd': [0,0,0,0,1,0,1,1],
    'e': [1,1,0,1,1,0,0,1],
    'label': [1,0,0,0,0,1,1,1],
    'weight': [1,1,1,1,1,1,1,1],
    'member': [1,0,0,0,0,1,1,1]}

df = pd.DataFrame(initial_data)
X = df[['a', 'b', 'c', 'd', 'e']].values
y = df['label'].values
clf = CustomDecisionTree(max_depth=6)
clf.fit(X, y, feature_names=df.columns[:-3])
predictions = clf.predict(X, expected_labels=y, weights=df['weight'].values, member=df['member'].values)
clf.save_tree(f'extwo_initial')
clf = MuteTree(clf, pd.DataFrame(data2), 0, 0, 1)
clf = MuteTree(clf, pd.DataFrame(data3), 0, 0, 2)
'''





'''
complete_data = {'a': [0,0,0,1,1,0,1,1,1,1,1,0,0,0,0,1],
                 'b': [0,0,0,0,0,1,0,1,1,1,1,0,1,1,1,0],
                 'c': [0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1],
                 'd': [0,1,0,0,1,1,1,0,1,0,1,1,0,1,0,0],
                 'label': [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0]
}

complete_df = pd.DataFrame(complete_data)
X = complete_df[['a', 'b', 'c', 'd']].values
y = complete_df['label'].values
complete_clf = CustomDecisionTree(max_depth=6)
complete_clf.fit(X, y, feature_names=complete_df.columns[:-1])
complete_clf.save_tree(output_file='complete_tree')



# member is 1 if datapoint is obtained from DistEstimate and 0 if is obtained from Validifier
data = {
    'a': [1, 1, 1, 0, 0, 1],
    'b': [1, 1, 1, 0, 1, 0],
    'c': [0, 0, 1, 0, 1, 0],
    'd': [0, 1, 0, 0, 0, 1],
    'label': [1, 1, 1, 0, 0, 0],
    'weight': [1, 1, 1, 1, 1, 1],
    'member': [1, 1, 1, 0, 0, 0]
}
data2 = {
    'a': [0,0,0,0,1],
    'b': [0,1,0,1,1],
    'c': [1,1,0,0,1],
    'd': [1,1,1,1,1],
    'label': [1,1,0,0,1],
    'weight': [1, 1, 1, 1, 1],
    'member': [1, 1, 0, 0, 0]
}


'''



'''###Program ex11 data

initial_dist_data = [(2, 1, 1), (5, 1, 1), (7, 1, 1), (10, 1, 1), (13, 1, 1), (15, 1, 1), (16, 1, 1), (18, 1, 1), (21, 1, 1), (23, 1, 1), (24, 1, 1), (26, 1, 1), (29, 1, 1), (64, 1, 1), (69, 1, 1), (71, 1, 1), (74, 1, 1), (77, 1, 1), (79, 1, 1), (80, 1, 1), (82, 1, 1), (85, 1, 1), (87, 1, 1), (88, 1, 1), (90, 1, 1), (95, 1, 1), (160, 1, 1), (162, 1, 1), (165, 1, 1), (167, 1, 1), (168, 1, 1), (170, 1, 1), (175, 1, 1), (176, 1, 1), (178, 1, 1), (181, 1, 1), (183, 1, 1), (184, 1, 1), (186, 1, 1), (189, 1, 1), (191, 1, 1), (224, 1, 1), (229, 1, 1), (231, 1, 1), (232, 1, 1), (234, 1, 1), (239, 1, 1), (240, 1, 1), (245, 1, 1), (248, 1, 1), (250, 1, 1), (253, 1, 1), (255, 1, 1)]
initial_valid_data = [(3, 1, 0), (6, 1, 0), (25, 1, 0), (27, 1, 0), (34, 1, 0), (38, 1, 0), (40, 1, 0), (46, 1, 0), (61, 1, 0), (65, 1, 0), (67, 1, 0), (81, 1, 0), (84, 1, 0), (98, 1, 0), (104, 1, 0), (110, 1, 0), (112, 1, 0), (114, 1, 0), (118, 1, 0), (126, 1, 0), (133, 1, 0), (136, 1, 0), (137, 1, 0), (138, 1, 0), (139, 1, 0), (150, 1, 0), (151, 1, 0), (152, 1, 0), (156, 1, 0), (157, 1, 0), (161, 1, 0), (174, 1, 0), (180, 1, 0), (185, 1, 0), (188, 1, 0), (190, 1, 0), (194, 1, 0), (206, 1, 0), (207, 1, 0), (208, 1, 0), (215, 1, 0), (217, 1, 0), (218, 1, 0), (221, 1, 0), (222, 1, 0), (223, 1, 0), (225, 1, 0), (228, 1, 0), (241, 1, 0), (246, 1, 0), (249, 1, 0), (252, 1, 0), (254, 1, 0)]
second_phase_dist_data = [(133, 1, 1), (55, 1, 1), (49, 1, 1), (113, 1, 1), (3, 1, 1), (33, 1, 1), (19, 1, 1), (37, 1, 1), (127, 1, 1), (215, 1, 1), (97, 1, 1), (39, 1, 1), (63, 1, 1), (101, 1, 1), (151, 1, 1), (125, 1, 1)]
second_phase_valid_data = [(163, 1, 0), (227, 1, 0), (235, 1, 0), (171, 1, 0), (187, 1, 0), (243, 1, 0), (179, 1, 0), (251, 1, 0)]


third_phase_dist_data = []
third_phase_valid_data = [(1, 1, 0), (6, 1, 0), (9, 1, 0), (11, 1, 0), (12, 1, 0), (14, 1, 0), (17, 1, 0), (20, 1, 0), (22, 1, 0), (25, 1, 0), (27, 1, 0), (28, 1, 0), (30, 1, 0), (32, 1, 0), (34, 1, 0), (35, 1, 0), (36, 1, 0), (38, 1, 0), (41, 1, 0), (42, 1, 0), (43, 1, 0), (44, 1, 0), (45, 1, 0), (46, 1, 0), (47, 1, 0), (48, 1, 0), (50, 1, 0), (51, 1, 0), (52, 1, 0), (53, 1, 0), (54, 1, 0), (56, 1, 0), (57, 1, 0), (58, 1, 0), (59, 1, 0), (60, 1, 0), (61, 1, 0), (62, 1, 0), (65, 1, 0), (67, 1, 0), (68, 1, 0), (73, 1, 0), (75, 1, 0), (76, 1, 0), (78, 1, 0), (81, 1, 0), (83, 1, 0), (84, 1, 0), (86, 1, 0), (89, 1, 0), (91, 1, 0), (92, 1, 0), (94, 1, 0), (98, 1, 0), (99, 1, 0), (100, 1, 0), (102, 1, 0), (103, 1, 0), (104, 1, 0), (105, 1, 0), (106, 1, 0), (107, 1, 0), (108, 1, 0), (109, 1, 0), (110, 1, 0), (111, 1, 0), (112, 1, 0), (116, 1, 0), (117, 1, 0), (118, 1, 0), (119, 1, 0), (120, 1, 0), (121, 1, 0), (122, 1, 0), (123, 1, 0), (124, 1, 0), (126, 1, 0), (128, 1, 0), (129, 1, 0), (130, 1, 0), (132, 1, 0), (134, 1, 0), (135, 1, 0), (136, 1, 0), (138, 1, 0), (140, 1, 0), (141, 1, 0), (143, 1, 0), (144, 1, 0), (145, 1, 0), (148, 1, 0), (149, 1, 0), (150, 1, 0), (152, 1, 0), (153, 1, 0), (154, 1, 0), (156, 1, 0), (157, 1, 0), (158, 1, 0), (159, 1, 0), (161, 1, 0), (164, 1, 0), (169, 1, 0), (172, 1, 0), (174, 1, 0), (177, 1, 0), (180, 1, 0), (182, 1, 0), (185, 1, 0), (188, 1, 0), (190, 1, 0), (192, 1, 0), (193, 1, 0), (194, 1, 0), (196, 1, 0), (197, 1, 0), (198, 1, 0), (199, 1, 0), (200, 1, 0), (201, 1, 0), (202, 1, 0), (204, 1, 0), (205, 1, 0), (207, 1, 0), (208, 1, 0), (209, 1, 0), (210, 1, 0), (212, 1, 0), (213, 1, 0), (214, 1, 0), (216, 1, 0), (217, 1, 0), (218, 1, 0), (220, 1, 0), (221, 1, 0), (222, 1, 0), (223, 1, 0), (225, 1, 0), (228, 1, 0), (230, 1, 0), (236, 1, 0), (238, 1, 0), (241, 1, 0), (244, 1, 0), (249, 1, 0), (252, 1, 0), (254, 1, 0)]

fourth_phase_dist_data = []
fourth_phase_valid_data = [(147,1,0),(211,1,0)]


initial_dist_data = initial_dist_data + second_phase_dist_data + third_phase_dist_data + fourth_phase_dist_data
initial_valid_data = initial_valid_data + second_phase_valid_data + third_phase_valid_data + fourth_phase_valid_data


second_phase_dist_data = [(213, 1, 1), (115, 1, 1), (211, 1, 1), (195, 1, 1), (197, 1, 1), (147, 1, 1), (131, 1, 1), (135, 1, 1)]
second_phase_valid_data = []
third_phase_dist_data = []
third_phase_valid_data = [(225, 1, 0), (129, 1, 0), (163, 1, 0), (99, 1, 0), (35, 1, 0), (227, 1, 0), (199, 1, 0), (193, 1, 0), (161, 1, 0), (145, 1, 0), (209, 1, 0), (51, 1, 0), (241, 1, 0), (177, 1, 0), (179, 1, 0)]

initial_dist_data = initial_dist_data + second_phase_dist_data + third_phase_dist_data 
initial_valid_data = initial_valid_data + second_phase_valid_data + third_phase_valid_data 


second_phase_dist_data = [(35,1,1),(163,1,1)]
second_phase_valid_data = []

third_phase_dist_data = []
third_phase_valid_data = [(243, 1, 0), (179, 1, 0), (227, 1, 0)]

fourth_phase_dist_data = []
fourth_phase_valid_data = [(51,1,0)]'''