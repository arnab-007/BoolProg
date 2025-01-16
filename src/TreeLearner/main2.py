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
    tree.save_tree(f'{PATH}/ex9_after_second_phase')
    distCurr, validCurr = error_bounds['dist_error'], error_bounds['valid_error']
    print(tree.print_leaf_datapoints())
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
        print("All moves:", [(move[0], move[3], move[4]) for move in moves])
        if not moves:
            raise ValueError("No valid moves found. Try increasing the max_split_count.")
        moves = Normalize(moves)
        moves.sort(key=lambda x: x[0], reverse=True)
        _, tree, error_bounds, leaf_id, mutation = random.choices(moves, weights=[move[0] for move in moves])[0]
        #print("leaf_id and mutation", leaf_id, mutation)
        tree.save_tree(f'{PATH}/intermediate/ex9_{leaf_id}_{mutation}')
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

from dd.autoref import BDD


def bdd_to_cnf(bdd, node):
    """Traverse BDD and extract CNF clauses."""
    if node == bdd.true:
        return [[]]  # CNF for True terminal is an empty clause
    if node == bdd.false:
        return None  # CNF for False terminal is an unsatisfiable clause

    var = node.var  # Get the variable name at the current node
    low_cnf = bdd_to_cnf(bdd, node.low)
    high_cnf = bdd_to_cnf(bdd, node.high)

    # Add the current variable to the CNF
    clauses = []
    if low_cnf:
        clauses += [[f"!{var}"] + clause for clause in low_cnf]  # Negate variable for low child
    if high_cnf:
        clauses += [[var] + clause for clause in high_cnf]  # Add variable for high child
    return clauses

def DNF_to_CNF_new(dnf_formula,variables):
    """Traverse BDD and extract CNF clauses."""
    bdd = BDD()
    bdd.declare(*variables)
    dnf_formula = (dnf_formula)


    # Replace '&&' with '&', '||' with '|', and adjust negations to '~'
    dnf_formula = (
        dnf_formula
        .replace("&&", "&")  # Replace AND
        .replace("||", "|")  # Replace OR
        .replace("!", "~")   # Replace NOT
    )

    # Wrap the output in parentheses and add line breaks for better readability
    dnf_formula = (
        "(" + 
        " | ".join(dnf_formula.split(" | ")) + 
        ")"
    )

    bdd_expr = bdd.add_expr(dnf_formula)
 
    cnf_clauses = bdd_to_cnf(bdd,bdd_expr)
    cnf_clauses = [f"({' || '.join(clause)})" for clause in cnf_clauses]
    # Join all clauses with '&&'
    cnf_formula = ' && '.join(cnf_clauses)
    return cnf_formula

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


#initial_dist_data = [(24, 1, 1), (42, 1, 1), (56, 1, 1), (96, 1, 1), (114, 1, 1), (294, 1, 1), (301, 1, 1), (310, 1, 1), (333, 1, 1), (375, 1, 1), (520, 1, 1), (530, 1, 1), (538, 1, 1), (545, 1, 1), (562, 1, 1), (571, 1, 1), (579, 1, 1), (593, 1, 1), (610, 1, 1), (618, 1, 1), (772, 1, 1), (782, 1, 1), (797, 1, 1), (804, 1, 1), (813, 1, 1), (822, 1, 1), (855, 1, 1), (1035, 1, 1), (1041, 1, 1), (1042, 1, 1), (1049, 1, 1), (1090, 1, 1), (1091, 1, 1), (1096, 1, 1), (1099, 1, 1), (1318, 1, 1), (1332, 1, 1), (1375, 1, 1), (1380, 1, 1), (1382, 1, 1), (1404, 1, 1), (1406, 1, 1), (1407, 1, 1), (1600, 1, 1), (1626, 1, 1), (1806, 1, 1), (1838, 1, 1), (1861, 1, 1), (1877, 1, 1), (1879, 1, 1), (1885, 1, 1), (1887, 1, 1), (2195, 1, 1), (2264, 1, 1), (2493, 1, 1), (2511, 1, 1), (2533, 1, 1), (2689, 1, 1), (2707, 1, 1), (2746, 1, 1), (2752, 1, 1), (2753, 1, 1), (2802, 1, 1), (2959, 1, 1), (2966, 1, 1), (2980, 1, 1), (2996, 1, 1), (3007, 1, 1), (3021, 1, 1), (3023, 1, 1), (3029, 1, 1), (3044, 1, 1), (3045, 1, 1), (3054, 1, 1), (3063, 1, 1), (3200, 1, 1), (3216, 1, 1), (3227, 1, 1), (3264, 1, 1), (3290, 1, 1), (3461, 1, 1), (3468, 1, 1), (3477, 1, 1), (3517, 1, 1), (3524, 1, 1), (3534, 1, 1), (3550, 1, 1), (3567, 1, 1), (3583, 1, 1), (3714, 1, 1), (3817, 1, 1), (3819, 1, 1), (3998, 1, 1), (4021, 1, 1), (4022, 1, 1), (4053, 1, 1), (4055, 1, 1), (4063, 1, 1)]
#initial_valid_data = [(6, 1, 0), (45, 1, 0), (147, 1, 0), (149, 1, 0), (314, 1, 0), (389, 1, 0), (404, 1, 0), (407, 1, 0), (445, 1, 0), (573, 1, 0), (591, 1, 0), (596, 1, 0), (604, 1, 0), (628, 1, 0), (636, 1, 0), (697, 1, 0), (701, 1, 0), (730, 1, 0), (747, 1, 0), (768, 1, 0), (808, 1, 0), (841, 1, 0), (865, 1, 0), (874, 1, 0), (898, 1, 0), (901, 1, 0), (956, 1, 0), (960, 1, 0), (980, 1, 0), (997, 1, 0), (1009, 1, 0), (1023, 1, 0), (1188, 1, 0), (1210, 1, 0), (1255, 1, 0), (1306, 1, 0), (1339, 1, 0), (1344, 1, 0), (1410, 1, 0), (1441, 1, 0), (1458, 1, 0), (1500, 1, 0), (1559, 1, 0), (1644, 1, 0), (1713, 1, 0), (1716, 1, 0), (1748, 1, 0), (1772, 1, 0), (1800, 1, 0), (1811, 1, 0), (1842, 1, 0), (1851, 1, 0), (1906, 1, 0), (1928, 1, 0), (2039, 1, 0), (2084, 1, 0), (2138, 1, 0), (2152, 1, 0), (2175, 1, 0), (2212, 1, 0), (2375, 1, 0), (2391, 1, 0), (2637, 1, 0), (2658, 1, 0), (2676, 1, 0), (2845, 1, 0), (2910, 1, 0), (2927, 1, 0), (2929, 1, 0), (2953, 1, 0), (2955, 1, 0), (3048, 1, 0), (3101, 1, 0), (3180, 1, 0), (3206, 1, 0), (3222, 1, 0), (3271, 1, 0), (3364, 1, 0), (3405, 1, 0), (3423, 1, 0), (3438, 1, 0), (3452, 1, 0), (3616, 1, 0), (3648, 1, 0), (3675, 1, 0), (3676, 1, 0), (3708, 1, 0), (3740, 1, 0), (3751, 1, 0), (3773, 1, 0), (3791, 1, 0), (3806, 1, 0), (3848, 1, 0), (3863, 1, 0), (3898, 1, 0), (3927, 1, 0), (3953, 1, 0), (4033, 1, 0)]
#second_phase_dist_data = [(16, 1, 1), (28, 1, 1), (26, 1, 1), (18, 1, 1)]
#second_phase_valid_data = [(2, 1, 0), (3, 1, 0), (4, 1, 0), (8, 1, 0), (9, 1, 0), (10, 1, 0), (11, 1, 0), (14, 1, 0), (15, 1, 0), (27, 1, 0), (29, 1, 0)]
initial_dist_data = [[0, 1, 1], [1, 1, 1], [6, 1, 1], [7, 1, 1], [24, 1, 1], [25, 1, 1], [30, 1, 1], [31, 1, 1]]
initial_valid_data = [[5, 1, 0], [9, 1, 0], [15, 1, 0], [16, 1, 0], [18, 1, 0], [21, 1, 0], [22, 1, 0], [27, 1, 0]]
#second_phase_dist_data = [(91, 1, 1), (88, 1, 1), (548, 1, 1), (580, 1, 1), (836, 1, 1), (804, 1, 1), (832, 1, 1), (760, 1, 1), (764, 1, 1), (800, 1, 1), (948, 1, 1), (551, 1, 1), (583, 1, 1), (839, 1, 1), (807, 1, 1), (52, 1, 1), (92, 1, 1), (48, 1, 1), (544, 1, 1), (576, 1, 1), (219, 1, 1), (220, 1, 1), (216, 1, 1), (951, 1, 1), (308, 1, 1), (436, 1, 1), (304, 1, 1), (944, 1, 1), (348, 1, 1), (345, 1, 1), (351, 1, 1), (432, 1, 1)]

#second_phase_valid_data = [(67, 1, 0), (69, 1, 0), (71, 1, 0), (72, 1, 0), (585, 1, 0), (74, 1, 0), (75, 1, 0), (76, 1, 0), (589, 1, 0), (78, 1, 0), (79, 1, 0), (591, 1, 0), (82, 1, 0), (86, 1, 0), (97, 1, 0), (101, 1, 0), (103, 1, 0), (104, 1, 0), (617, 1, 0), (619, 1, 0), (108, 1, 0), (109, 1, 0), (107, 1, 0), (111, 1, 0), (623, 1, 0), (110, 1, 0), (621, 1, 0), (114, 1, 0), (118, 1, 0), (195, 1, 0), (199, 1, 0), (200, 1, 0), (201, 1, 0), (713, 1, 0), (715, 1, 0), (205, 1, 0), (717, 1, 0), (206, 1, 0), (207, 1, 0), (719, 1, 0), (210, 1, 0), (214, 1, 0), (225, 1, 0), (227, 1, 0), (229, 1, 0), (231, 1, 0), (232, 1, 0), (745, 1, 0), (235, 1, 0), (236, 1, 0), (237, 1, 0), (238, 1, 0), (751, 1, 0), (239, 1, 0), (242, 1, 0), (246, 1, 0), (321, 1, 0), (325, 1, 0), (328, 1, 0), (841, 1, 0), (330, 1, 0), (331, 1, 0), (843, 1, 0), (333, 1, 0), (332, 1, 0), (847, 1, 0), (338, 1, 0), (353, 1, 0), (357, 1, 0), (360, 1, 0), (873, 1, 0), (362, 1, 0), (363, 1, 0), (875, 1, 0), (877, 1, 0), (364, 1, 0), (367, 1, 0), (879, 1, 0), (370, 1, 0), (374, 1, 0), (376, 1, 0), (378, 1, 0), (449, 1, 0), (456, 1, 0), (457, 1, 0), (458, 1, 0), (460, 1, 0), (462, 1, 0), (466, 1, 0), (481, 1, 0), (485, 1, 0), (488, 1, 0), (491, 1, 0), (495, 1, 0), (502, 1, 0)]

number_of_vars = 5
df_dist = prepare_data(initial_dist_data, number_of_vars, DistEstimate = True)
df_valid = prepare_data(initial_valid_data, number_of_vars, DistEstimate = False)
df = pd.concat([df_dist, df_valid], ignore_index=True)
X = df[list(df.columns[:-3])].values
y = df['label'].values
clf = CustomDecisionTree(max_depth= number_of_vars)
clf.fit_initial(X, y, feature_names=df.columns[:-3])
#print("initial dnf", clf.tree_to_dnf())
clf.save_tree(f'{PATH}/ex9_1')
predictions = clf.predict(df_dist[list(df.columns[:-3])].values, expected_labels=df_dist['label'].values, weights=df_dist['weight'].values, member=df_dist['member'].values)
print("Initial Bounds(before prediction):", clf.get_error_bounds())
clf.save_tree(f'{PATH}/ex9_2')
print("DNF:", clf.tree_to_dnf())
print("CNF by sympy:", DNF_to_CNF(clf.tree_to_dnf()))
print("CNF by BDD:", DNF_to_CNF_new(clf.tree_to_dnf(),[
            "x1",
            "x2",
            "x3",
            "x4",
            "x5"
        ]))
#print("dnf", clf.tree_to_dnf())
#df_dist_second = prepare_data(second_phase_dist_data, number_of_vars, DistEstimate = True)
#df_valid_second = prepare_data(second_phase_valid_data, number_of_vars, DistEstimate = False)
#df_second = pd.concat([df_dist_second, df_valid_second], ignore_index=True)
#clf.predict(df_second[list(df.columns[:-3])].values, expected_labels=df_second['label'].values, weights=df_second['weight'].values, member=df_second['member'].values)
#print("Initial Bounds(after prediction):", clf.get_error_bounds())

# df_dist_third = prepare_data(third_phase_dist_data, number_of_vars, DistEstimate = True)
# df_valid_third = prepare_data(third_phase_valid_data, number_of_vars, DistEstimate = False)
# df_third = pd.concat([df_dist_third, df_valid_third], ignore_index=True)

# df_dist_fourth = prepare_data(fourth_phase_dist_data, number_of_vars, DistEstimate = True)
# df_valid_fourth = prepare_data(fourth_phase_valid_data, number_of_vars, DistEstimate = False)
# df_fourth = pd.concat([df_dist_fourth, df_valid_fourth], ignore_index=True)


'''

for iteration in range(4,5):
    clf_copy = copy.deepcopy(clf)
    print(f'{iteration=}')
    random.seed(iteration + 10)
    print("Second Phase")
    final_tree, tree_sequence = MuteTree(clf_copy, df_second, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex9_second_1')
    print("CNF:", DNF_to_CNF(final_tree.tree_to_dnf()))

'''





