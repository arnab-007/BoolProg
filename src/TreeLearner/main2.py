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
    max_split_count = 1000000
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
    print("Next Phase Bounds:", error_bounds)
    tree.save_tree(f'{PATH}/ex9_after_second_phase')
    distCurr, validCurr = error_bounds['dist_error'], error_bounds['valid_error']
    #print(tree.print_leaf_datapoints())
    set_of_mutations = ['split', 'prune']
    split_count_map = get_split_count_map(tree)
    tree_sequence = {}
    while(time.time() - start < 500):
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
            print(tree.print_leaf_datapoints())
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
            print("Optimal Tree Error Bounds:", tree.get_error_bounds())
            return tree, tree_sequence
    '''If timeout occurs, we return the tree with the best error bounds so far by sorting the dictionary tree_sequence and extracting the first element.'''
    print("Timeout occurred. Returning the best tree so far.")
    print(tree.print_leaf_datapoints())
    # best_tree = sorted(tree_sequence.items(), key=lambda x: round(np.sqrt(x[1]['dist_error'] + x[1]['valid_error'])))[0][0]
    best_tree = sorted(tree_sequence.items(), key=lambda x: round(np.sqrt(x[1]['dist_error'] + x[1]['valid_error']), 2))[0][0]
    print("Best Tree Error Bounds:", best_tree.get_error_bounds())
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


#from z3 import *



def remove_tautologies(cnf_formula):
    def is_tautology(clause):
        # Check if a clause contains both a literal and its negation
        literals = set(clause)
        for literal in literals:
            if literal.startswith("!") and literal[1:] in literals:
                return True
        return False
    
    # Filter out the clauses that are tautologies
    filtered_formula = [clause for clause in cnf_formula if not is_tautology(clause)]
    return filtered_formula  # Make sure to return the filtered formula


def cnf_to_string(cnf_formula):
    cnf_string = " || ".join(
        "(" + " && ".join(clause) + ")" for clause in cnf_formula
    )
    return cnf_string
    
    
def DNF_to_CNF(dnf_formula):
    cnf_formula = []

    # If DNF formula is empty, return empty CNF formula
    if not dnf_formula:
        return cnf_formula

    # Initialize CNF formula with the first clause of DNF
    cnf_formula = [[literal] for literal in dnf_formula[0]]
    
    # Distribute each subsequent clause over the existing CNF formula
    for i in range(1, len(dnf_formula)):
        new_cnf_formula = []
        for clause in cnf_formula:
            for literal in dnf_formula[i]:
                
                new_clause = clause + [literal]
                
                new_cnf_formula.append(list(set(new_clause)))
        cnf_formula = new_cnf_formula

    # Remove duplicate literals (both x and -x) within the same clause
    #cnf_formula = [remove_duplicate_literals(clause) for clause in cnf_formula]
    cnf_formula = [elem for elem in cnf_formula if elem]
    unique_set = {tuple(sorted(sublist)) for sublist in cnf_formula}
    cnf_formula = [list(tup) for tup in unique_set]
    
    return cnf_to_string(remove_tautologies(cnf_formula))



#initial_dist_data = [(24, 1, 1), (42, 1, 1), (56, 1, 1), (96, 1, 1), (114, 1, 1), (294, 1, 1), (301, 1, 1), (310, 1, 1), (333, 1, 1), (375, 1, 1), (520, 1, 1), (530, 1, 1), (538, 1, 1), (545, 1, 1), (562, 1, 1), (571, 1, 1), (579, 1, 1), (593, 1, 1), (610, 1, 1), (618, 1, 1), (772, 1, 1), (782, 1, 1), (797, 1, 1), (804, 1, 1), (813, 1, 1), (822, 1, 1), (855, 1, 1), (1035, 1, 1), (1041, 1, 1), (1042, 1, 1), (1049, 1, 1), (1090, 1, 1), (1091, 1, 1), (1096, 1, 1), (1099, 1, 1), (1318, 1, 1), (1332, 1, 1), (1375, 1, 1), (1380, 1, 1), (1382, 1, 1), (1404, 1, 1), (1406, 1, 1), (1407, 1, 1), (1600, 1, 1), (1626, 1, 1), (1806, 1, 1), (1838, 1, 1), (1861, 1, 1), (1877, 1, 1), (1879, 1, 1), (1885, 1, 1), (1887, 1, 1), (2195, 1, 1), (2264, 1, 1), (2493, 1, 1), (2511, 1, 1), (2533, 1, 1), (2689, 1, 1), (2707, 1, 1), (2746, 1, 1), (2752, 1, 1), (2753, 1, 1), (2802, 1, 1), (2959, 1, 1), (2966, 1, 1), (2980, 1, 1), (2996, 1, 1), (3007, 1, 1), (3021, 1, 1), (3023, 1, 1), (3029, 1, 1), (3044, 1, 1), (3045, 1, 1), (3054, 1, 1), (3063, 1, 1), (3200, 1, 1), (3216, 1, 1), (3227, 1, 1), (3264, 1, 1), (3290, 1, 1), (3461, 1, 1), (3468, 1, 1), (3477, 1, 1), (3517, 1, 1), (3524, 1, 1), (3534, 1, 1), (3550, 1, 1), (3567, 1, 1), (3583, 1, 1), (3714, 1, 1), (3817, 1, 1), (3819, 1, 1), (3998, 1, 1), (4021, 1, 1), (4022, 1, 1), (4053, 1, 1), (4055, 1, 1), (4063, 1, 1)]
#initial_valid_data = [(6, 1, 0), (45, 1, 0), (147, 1, 0), (149, 1, 0), (314, 1, 0), (389, 1, 0), (404, 1, 0), (407, 1, 0), (445, 1, 0), (573, 1, 0), (591, 1, 0), (596, 1, 0), (604, 1, 0), (628, 1, 0), (636, 1, 0), (697, 1, 0), (701, 1, 0), (730, 1, 0), (747, 1, 0), (768, 1, 0), (808, 1, 0), (841, 1, 0), (865, 1, 0), (874, 1, 0), (898, 1, 0), (901, 1, 0), (956, 1, 0), (960, 1, 0), (980, 1, 0), (997, 1, 0), (1009, 1, 0), (1023, 1, 0), (1188, 1, 0), (1210, 1, 0), (1255, 1, 0), (1306, 1, 0), (1339, 1, 0), (1344, 1, 0), (1410, 1, 0), (1441, 1, 0), (1458, 1, 0), (1500, 1, 0), (1559, 1, 0), (1644, 1, 0), (1713, 1, 0), (1716, 1, 0), (1748, 1, 0), (1772, 1, 0), (1800, 1, 0), (1811, 1, 0), (1842, 1, 0), (1851, 1, 0), (1906, 1, 0), (1928, 1, 0), (2039, 1, 0), (2084, 1, 0), (2138, 1, 0), (2152, 1, 0), (2175, 1, 0), (2212, 1, 0), (2375, 1, 0), (2391, 1, 0), (2637, 1, 0), (2658, 1, 0), (2676, 1, 0), (2845, 1, 0), (2910, 1, 0), (2927, 1, 0), (2929, 1, 0), (2953, 1, 0), (2955, 1, 0), (3048, 1, 0), (3101, 1, 0), (3180, 1, 0), (3206, 1, 0), (3222, 1, 0), (3271, 1, 0), (3364, 1, 0), (3405, 1, 0), (3423, 1, 0), (3438, 1, 0), (3452, 1, 0), (3616, 1, 0), (3648, 1, 0), (3675, 1, 0), (3676, 1, 0), (3708, 1, 0), (3740, 1, 0), (3751, 1, 0), (3773, 1, 0), (3791, 1, 0), (3806, 1, 0), (3848, 1, 0), (3863, 1, 0), (3898, 1, 0), (3927, 1, 0), (3953, 1, 0), (4033, 1, 0)]
#second_phase_dist_data = [(16, 1, 1), (28, 1, 1), (26, 1, 1), (18, 1, 1)]
#second_phase_valid_data = [(2, 1, 0), (3, 1, 0), (4, 1, 0), (8, 1, 0), (9, 1, 0), (10, 1, 0), (11, 1, 0), (14, 1, 0), (15, 1, 0), (27, 1, 0), (29, 1, 0)]
initial_dist_data = [(19, 1, 1), (23, 1, 1), (36, 1, 1), (38, 1, 1), (53, 1, 1), (55, 1, 1), (70, 1, 1), (83, 1, 1), (87, 1, 1), (96, 1, 1), (98, 1, 1), (113, 1, 1), (117, 1, 1), (128, 1, 1), (149, 1, 1), (160, 1, 1), (162, 1, 1), (164, 1, 1), (166, 1, 1), (179, 1, 1), (183, 1, 1), (215, 1, 1), (241, 1, 1), (260, 1, 1), (273, 1, 1), (275, 1, 1), (277, 1, 1), (294, 1, 1), (305, 1, 1), (309, 1, 1), (311, 1, 1), (337, 1, 1), (341, 1, 1), (352, 1, 1), (354, 1, 1), (358, 1, 1), (403, 1, 1), (407, 1, 1), (418, 1, 1), (422, 1, 1), (433, 1, 1), (437, 1, 1), (448, 1, 1), (469, 1, 1), (482, 1, 1), (499, 1, 1), (526, 1, 1), (558, 1, 1), (575, 1, 1), (584, 1, 1), (601, 1, 1), (603, 1, 1), (616, 1, 1), (622, 1, 1), (648, 1, 1), (669, 1, 1), (680, 1, 1), (686, 1, 1), (697, 1, 1), (703, 1, 1), (714, 1, 1), (748, 1, 1), (750, 1, 1), (780, 1, 1), (797, 1, 1), (829, 1, 1), (859, 1, 1), (861, 1, 1), (872, 1, 1), (891, 1, 1), (893, 1, 1), (910, 1, 1), (921, 1, 1), (936, 1, 1), (942, 1, 1), (955, 1, 1), (985, 1, 1), (987, 1, 1), (1002, 1, 1), (1017, 1, 1), (1019, 1, 1)]

initial_valid_data = [(29, 1, 0), (59, 1, 0), (61, 1, 0), (62, 1, 0), (74, 1, 0), (82, 1, 0), (86, 1, 0), (106, 1, 0), (109, 1, 0), (112, 1, 0), (118, 1, 0), (161, 1, 0), (172, 1, 0), (185, 1, 0), (186, 1, 0), (191, 1, 0), (193, 1, 0), (205, 1, 0), (222, 1, 0), (248, 1, 0), (271, 1, 0), (281, 1, 0), (282, 1, 0), (314, 1, 0), (317, 1, 0), (323, 1, 0), (342, 1, 0), (344, 1, 0), (353, 1, 0), (361, 1, 0), (378, 1, 0), (379, 1, 0), (385, 1, 0), (419, 1, 0), (434, 1, 0), (440, 1, 0), (468, 1, 0), (474, 1, 0), (478, 1, 0), (483, 1, 0), (508, 1, 0), (513, 1, 0), (523, 1, 0), (559, 1, 0), (565, 1, 0), (579, 1, 0), (585, 1, 0), (599, 1, 0), (602, 1, 0), (614, 1, 0), (642, 1, 0), (659, 1, 0), (673, 1, 0), (679, 1, 0), (685, 1, 0), (713, 1, 0), (728, 1, 0), (730, 1, 0), (732, 1, 0), (739, 1, 0), (742, 1, 0), (749, 1, 0), (752, 1, 0), (791, 1, 0), (809, 1, 0), (815, 1, 0), (821, 1, 0), (823, 1, 0), (835, 1, 0), (845, 1, 0), (858, 1, 0), (879, 1, 0), (903, 1, 0), (915, 1, 0), (933, 1, 0), (935, 1, 0), (947, 1, 0), (975, 1, 0), (978, 1, 0), (1008, 1, 0), (1022, 1, 0)]

second_phase_dist_data = [(55, 1, 1), (1017, 1, 1), (763, 1, 1), (311, 1, 1)]

second_phase_valid_data = [(0, 1, 0), (2, 1, 0), (4, 1, 0), (6, 1, 0), (519, 1, 0), (9, 1, 0), (521, 1, 0), (12, 1, 0), (525, 1, 0), (16, 1, 0), (18, 1, 0), (20, 1, 0), (22, 1, 0), (24, 1, 0), (27, 1, 0), (545, 1, 0), (547, 1, 0), (549, 1, 0), (40, 1, 0), (41, 1, 0), (42, 1, 0), (555, 1, 0), (557, 1, 0), (46, 1, 0), (50, 1, 0), (52, 1, 0), (565, 1, 0), (54, 1, 0), (567, 1, 0), (56, 1, 0), (58, 1, 0), (59, 1, 0), (61, 1, 0), (577, 1, 0), (579, 1, 0), (581, 1, 0), (75, 1, 0), (76, 1, 0), (77, 1, 0), (589, 1, 0), (591, 1, 0), (593, 1, 0), (82, 1, 0), (595, 1, 0), (86, 1, 0), (599, 1, 0), (88, 1, 0), (89, 1, 0), (90, 1, 0), (93, 1, 0), (94, 1, 0), (609, 1, 0), (615, 1, 0), (104, 1, 0), (105, 1, 0), (619, 1, 0), (109, 1, 0), (111, 1, 0), (112, 1, 0), (623, 1, 0), (114, 1, 0), (118, 1, 0), (120, 1, 0), (121, 1, 0), (122, 1, 0), (123, 1, 0), (124, 1, 0), (125, 1, 0), (126, 1, 0), (641, 1, 0), (645, 1, 0), (134, 1, 0), (647, 1, 0), (649, 1, 0), (139, 1, 0), (140, 1, 0), (651, 1, 0), (653, 1, 0), (655, 1, 0), (144, 1, 0), (657, 1, 0), (148, 1, 0), (661, 1, 0), (150, 1, 0), (153, 1, 0), (154, 1, 0), (155, 1, 0), (158, 1, 0), (675, 1, 0), (679, 1, 0), (169, 1, 0), (681, 1, 0), (171, 1, 0), (172, 1, 0), (173, 1, 0), (174, 1, 0), (175, 1, 0), (176, 1, 0), (683, 1, 0), (687, 1, 0), (689, 1, 0), (180, 1, 0), (691, 1, 0), (182, 1, 0), (693, 1, 0), (184, 1, 0), (185, 1, 0), (187, 1, 0), (188, 1, 0), (189, 1, 0), (191, 1, 0), (705, 1, 0), (707, 1, 0), (709, 1, 0), (198, 1, 0), (711, 1, 0), (201, 1, 0), (202, 1, 0), (203, 1, 0), (713, 1, 0), (205, 1, 0), (206, 1, 0), (717, 1, 0), (208, 1, 0), (719, 1, 0), (210, 1, 0), (723, 1, 0), (212, 1, 0), (214, 1, 0), (727, 1, 0), (216, 1, 0), (218, 1, 0), (220, 1, 0), (221, 1, 0), (222, 1, 0), (224, 1, 0), (737, 1, 0), (739, 1, 0), (228, 1, 0), (741, 1, 0), (743, 1, 0), (232, 1, 0), (240, 1, 0), (753, 1, 0), (242, 1, 0), (759, 1, 0), (250, 1, 0), (251, 1, 0), (253, 1, 0), (771, 1, 0), (773, 1, 0), (262, 1, 0), (775, 1, 0), (266, 1, 0), (779, 1, 0), (268, 1, 0), (269, 1, 0), (270, 1, 0), (272, 1, 0), (785, 1, 0), (274, 1, 0), (787, 1, 0), (276, 1, 0), (278, 1, 0), (791, 1, 0), (280, 1, 0), (282, 1, 0), (286, 1, 0), (287, 1, 0), (290, 1, 0), (803, 1, 0), (296, 1, 0), (297, 1, 0), (298, 1, 0), (809, 1, 0), (300, 1, 0), (301, 1, 0), (302, 1, 0), (303, 1, 0), (304, 1, 0), (811, 1, 0), (306, 1, 0), (813, 1, 0), (815, 1, 0), (817, 1, 0), (310, 1, 0), (821, 1, 0), (312, 1, 0), (823, 1, 0), (314, 1, 0), (315, 1, 0), (318, 1, 0), (319, 1, 0), (835, 1, 0), (837, 1, 0), (326, 1, 0), (841, 1, 0), (332, 1, 0), (335, 1, 0), (336, 1, 0), (847, 1, 0), (338, 1, 0), (851, 1, 0), (340, 1, 0), (344, 1, 0), (350, 1, 0), (865, 1, 0), (869, 1, 0), (873, 1, 0), (362, 1, 0), (363, 1, 0), (364, 1, 0), (877, 1, 0), (366, 1, 0), (368, 1, 0), (881, 1, 0), (370, 1, 0), (372, 1, 0), (885, 1, 0), (374, 1, 0), (376, 1, 0), (378, 1, 0), (379, 1, 0), (380, 1, 0), (382, 1, 0), (384, 1, 0), (897, 1, 0), (386, 1, 0), (899, 1, 0), (388, 1, 0), (901, 1, 0), (392, 1, 0), (393, 1, 0), (394, 1, 0), (905, 1, 0), (396, 1, 0), (907, 1, 0), (398, 1, 0), (400, 1, 0), (913, 1, 0), (404, 1, 0), (406, 1, 0), (408, 1, 0), (415, 1, 0), (416, 1, 0), (929, 1, 0), (931, 1, 0), (420, 1, 0), (424, 1, 0), (426, 1, 0), (427, 1, 0), (428, 1, 0), (941, 1, 0), (430, 1, 0), (431, 1, 0), (945, 1, 0), (434, 1, 0), (949, 1, 0), (438, 1, 0), (444, 1, 0), (445, 1, 0), (961, 1, 0), (452, 1, 0), (965, 1, 0), (456, 1, 0), (457, 1, 0), (458, 1, 0), (969, 1, 0), (973, 1, 0), (463, 1, 0), (464, 1, 0), (977, 1, 0), (979, 1, 0), (983, 1, 0), (472, 1, 0), (473, 1, 0), (474, 1, 0), (476, 1, 0), (477, 1, 0), (478, 1, 0), (480, 1, 0), (993, 1, 0), (484, 1, 0), (486, 1, 0), (999, 1, 0), (488, 1, 0), (489, 1, 0), (490, 1, 0), (491, 1, 0), (1001, 1, 0), (1007, 1, 0), (496, 1, 0), (498, 1, 0), (1011, 1, 0), (500, 1, 0), (502, 1, 0), (1015, 1, 0), (504, 1, 0), (505, 1, 0), (506, 1, 0), (507, 1, 0), (509, 1, 0), (511, 1, 0)]

third_phase_valid_data = [(0, 1, 0), (513, 1, 0), (2, 1, 0), (515, 1, 0), (4, 1, 0), (517, 1, 0), (8, 1, 0), (521, 1, 0), (10, 1, 0), (523, 1, 0), (12, 1, 0), (13, 1, 0), (14, 1, 0), (525, 1, 0), (16, 1, 0), (529, 1, 0), (18, 1, 0), (20, 1, 0), (22, 1, 0), (535, 1, 0), (28, 1, 0), (31, 1, 0), (547, 1, 0), (40, 1, 0), (41, 1, 0), (42, 1, 0), (43, 1, 0), (44, 1, 0), (555, 1, 0), (557, 1, 0), (50, 1, 0), (563, 1, 0), (52, 1, 0), (565, 1, 0), (54, 1, 0), (56, 1, 0), (57, 1, 0), (59, 1, 0), (577, 1, 0), (66, 1, 0), (73, 1, 0), (74, 1, 0), (75, 1, 0), (585, 1, 0), (589, 1, 0), (79, 1, 0), (80, 1, 0), (82, 1, 0), (595, 1, 0), (84, 1, 0), (86, 1, 0), (599, 1, 0), (89, 1, 0), (90, 1, 0), (93, 1, 0), (94, 1, 0), (95, 1, 0), (613, 1, 0), (615, 1, 0), (104, 1, 0), (106, 1, 0), (619, 1, 0), (621, 1, 0), (623, 1, 0), (625, 1, 0), (118, 1, 0), (120, 1, 0), (122, 1, 0), (123, 1, 0), (125, 1, 0), (126, 1, 0), (641, 1, 0), (130, 1, 0), (132, 1, 0), (134, 1, 0), (647, 1, 0), (136, 1, 0), (137, 1, 0), (649, 1, 0), (140, 1, 0), (653, 1, 0), (655, 1, 0), (657, 1, 0), (148, 1, 0), (661, 1, 0), (150, 1, 0), (152, 1, 0), (153, 1, 0), (155, 1, 0), (157, 1, 0), (159, 1, 0), (677, 1, 0), (679, 1, 0), (170, 1, 0), (171, 1, 0), (683, 1, 0), (685, 1, 0), (174, 1, 0), (687, 1, 0), (689, 1, 0), (178, 1, 0), (180, 1, 0), (693, 1, 0), (182, 1, 0), (185, 1, 0), (188, 1, 0), (189, 1, 0), (191, 1, 0), (192, 1, 0), (705, 1, 0), (194, 1, 0), (709, 1, 0), (200, 1, 0), (713, 1, 0), (204, 1, 0), (717, 1, 0), (206, 1, 0), (207, 1, 0), (208, 1, 0), (721, 1, 0), (210, 1, 0), (723, 1, 0), (212, 1, 0), (727, 1, 0), (216, 1, 0), (217, 1, 0), (220, 1, 0), (222, 1, 0), (223, 1, 0), (224, 1, 0), (737, 1, 0), (228, 1, 0), (743, 1, 0), (745, 1, 0), (234, 1, 0), (747, 1, 0), (237, 1, 0), (749, 1, 0), (239, 1, 0), (240, 1, 0), (751, 1, 0), (753, 1, 0), (755, 1, 0), (244, 1, 0), (757, 1, 0), (251, 1, 0), (253, 1, 0), (254, 1, 0), (256, 1, 0), (769, 1, 0), (773, 1, 0), (262, 1, 0), (775, 1, 0), (265, 1, 0), (266, 1, 0), (777, 1, 0), (779, 1, 0), (269, 1, 0), (781, 1, 0), (272, 1, 0), (274, 1, 0), (787, 1, 0), (278, 1, 0), (282, 1, 0), (285, 1, 0), (287, 1, 0), (290, 1, 0), (803, 1, 0), (296, 1, 0), (298, 1, 0), (299, 1, 0), (300, 1, 0), (813, 1, 0), (302, 1, 0), (306, 1, 0), (819, 1, 0), (310, 1, 0), (823, 1, 0), (314, 1, 0), (319, 1, 0), (322, 1, 0), (837, 1, 0), (326, 1, 0), (328, 1, 0), (329, 1, 0), (331, 1, 0), (843, 1, 0), (845, 1, 0), (334, 1, 0), (847, 1, 0), (849, 1, 0), (851, 1, 0), (340, 1, 0), (853, 1, 0), (342, 1, 0), (344, 1, 0), (346, 1, 0), (347, 1, 0), (348, 1, 0), (350, 1, 0), (865, 1, 0), (867, 1, 0), (356, 1, 0), (360, 1, 0), (361, 1, 0), (362, 1, 0), (363, 1, 0), (877, 1, 0), (366, 1, 0), (879, 1, 0), (368, 1, 0), (881, 1, 0), (370, 1, 0), (883, 1, 0), (372, 1, 0), (374, 1, 0), (887, 1, 0), (378, 1, 0), (380, 1, 0), (381, 1, 0), (382, 1, 0), (383, 1, 0), (384, 1, 0), (386, 1, 0), (899, 1, 0), (390, 1, 0), (903, 1, 0), (393, 1, 0), (395, 1, 0), (396, 1, 0), (907, 1, 0), (398, 1, 0), (399, 1, 0), (400, 1, 0), (911, 1, 0), (913, 1, 0), (915, 1, 0), (917, 1, 0), (919, 1, 0), (409, 1, 0), (410, 1, 0), (412, 1, 0), (416, 1, 0), (929, 1, 0), (931, 1, 0), (933, 1, 0), (424, 1, 0), (425, 1, 0), (426, 1, 0), (427, 1, 0), (428, 1, 0), (429, 1, 0), (430, 1, 0), (431, 1, 0), (939, 1, 0), (941, 1, 0), (949, 1, 0), (438, 1, 0), (442, 1, 0), (443, 1, 0), (444, 1, 0), (446, 1, 0), (447, 1, 0), (450, 1, 0), (963, 1, 0), (967, 1, 0), (456, 1, 0), (457, 1, 0), (458, 1, 0), (459, 1, 0), (460, 1, 0), (969, 1, 0), (462, 1, 0), (463, 1, 0), (973, 1, 0), (975, 1, 0), (466, 1, 0), (979, 1, 0), (981, 1, 0), (470, 1, 0), (472, 1, 0), (473, 1, 0), (474, 1, 0), (479, 1, 0), (993, 1, 0), (486, 1, 0), (999, 1, 0), (490, 1, 0), (491, 1, 0), (494, 1, 0), (495, 1, 0), (496, 1, 0), (498, 1, 0), (1011, 1, 0), (500, 1, 0), (1013, 1, 0), (502, 1, 0), (1015, 1, 0), (504, 1, 0), (507, 1, 0), (509, 1, 0), (510, 1, 0)]

third_phase_dist_data = [(55, 1, 1), (311, 1, 1), (763, 1, 1)]

fourth_phase_dist_data = [(763, 1, 1), (55, 1, 1), (439, 1, 1), (311, 1, 1)]

fourth_phase_valid_data = [(0, 1, 0), (4, 1, 0), (8, 1, 0), (521, 1, 0), (10, 1, 0), (12, 1, 0), (13, 1, 0), (14, 1, 0), (15, 1, 0), (527, 1, 0), (525, 1, 0), (18, 1, 0), (20, 1, 0), (533, 1, 0), (22, 1, 0), (535, 1, 0), (24, 1, 0), (27, 1, 0), (28, 1, 0), (31, 1, 0), (545, 1, 0), (549, 1, 0), (40, 1, 0), (553, 1, 0), (41, 1, 0), (43, 1, 0), (44, 1, 0), (45, 1, 0), (42, 1, 0), (47, 1, 0), (48, 1, 0), (561, 1, 0), (50, 1, 0), (563, 1, 0), (565, 1, 0), (54, 1, 0), (56, 1, 0), (57, 1, 0), (59, 1, 0), (60, 1, 0), (61, 1, 0), (577, 1, 0), (72, 1, 0), (73, 1, 0), (75, 1, 0), (78, 1, 0), (591, 1, 0), (80, 1, 0), (82, 1, 0), (595, 1, 0), (84, 1, 0), (86, 1, 0), (90, 1, 0), (92, 1, 0), (611, 1, 0), (100, 1, 0), (102, 1, 0), (615, 1, 0), (105, 1, 0), (106, 1, 0), (617, 1, 0), (109, 1, 0), (110, 1, 0), (111, 1, 0), (621, 1, 0), (625, 1, 0), (116, 1, 0), (629, 1, 0), (118, 1, 0), (123, 1, 0), (641, 1, 0), (130, 1, 0), (643, 1, 0), (132, 1, 0), (645, 1, 0), (134, 1, 0), (647, 1, 0), (136, 1, 0), (649, 1, 0), (138, 1, 0), (137, 1, 0), (651, 1, 0), (141, 1, 0), (655, 1, 0), (144, 1, 0), (657, 1, 0), (146, 1, 0), (659, 1, 0), (661, 1, 0), (150, 1, 0), (153, 1, 0), (155, 1, 0), (157, 1, 0), (158, 1, 0), (159, 1, 0), (673, 1, 0), (170, 1, 0), (683, 1, 0), (171, 1, 0), (174, 1, 0), (687, 1, 0), (176, 1, 0), (689, 1, 0), (178, 1, 0), (691, 1, 0), (180, 1, 0), (182, 1, 0), (695, 1, 0), (184, 1, 0), (187, 1, 0), (191, 1, 0), (705, 1, 0), (196, 1, 0), (198, 1, 0), (713, 1, 0), (201, 1, 0), (715, 1, 0), (202, 1, 0), (205, 1, 0), (204, 1, 0), (206, 1, 0), (208, 1, 0), (717, 1, 0), (210, 1, 0), (207, 1, 0), (212, 1, 0), (721, 1, 0), (214, 1, 0), (727, 1, 0), (723, 1, 0), (217, 1, 0), (218, 1, 0), (220, 1, 0), (222, 1, 0), (223, 1, 0), (226, 1, 0), (745, 1, 0), (233, 1, 0), (235, 1, 0), (236, 1, 0), (747, 1, 0), (237, 1, 0), (240, 1, 0), (753, 1, 0), (242, 1, 0), (244, 1, 0), (246, 1, 0), (248, 1, 0), (251, 1, 0), (252, 1, 0), (254, 1, 0), (255, 1, 0), (771, 1, 0), (773, 1, 0), (775, 1, 0), (264, 1, 0), (265, 1, 0), (777, 1, 0), (267, 1, 0), (779, 1, 0), (781, 1, 0), (270, 1, 0), (783, 1, 0), (271, 1, 0), (785, 1, 0), (268, 1, 0), (274, 1, 0), (269, 1, 0), (789, 1, 0), (278, 1, 0), (791, 1, 0), (276, 1, 0), (283, 1, 0), (284, 1, 0), (285, 1, 0), (286, 1, 0), (805, 1, 0), (296, 1, 0), (297, 1, 0), (298, 1, 0), (300, 1, 0), (813, 1, 0), (302, 1, 0), (815, 1, 0), (304, 1, 0), (817, 1, 0), (303, 1, 0), (819, 1, 0), (306, 1, 0), (821, 1, 0), (310, 1, 0), (314, 1, 0), (316, 1, 0), (317, 1, 0), (837, 1, 0), (326, 1, 0), (841, 1, 0), (329, 1, 0), (331, 1, 0), (332, 1, 0), (330, 1, 0), (334, 1, 0), (847, 1, 0), (336, 1, 0), (849, 1, 0), (338, 1, 0), (335, 1, 0), (853, 1, 0), (342, 1, 0), (344, 1, 0), (346, 1, 0), (347, 1, 0), (350, 1, 0), (867, 1, 0), (356, 1, 0), (869, 1, 0), (362, 1, 0), (875, 1, 0), (363, 1, 0), (366, 1, 0), (368, 1, 0), (370, 1, 0), (883, 1, 0), (372, 1, 0), (887, 1, 0), (377, 1, 0), (378, 1, 0), (379, 1, 0), (380, 1, 0), (382, 1, 0), (384, 1, 0), (386, 1, 0), (388, 1, 0), (390, 1, 0), (903, 1, 0), (907, 1, 0), (396, 1, 0), (909, 1, 0), (911, 1, 0), (399, 1, 0), (400, 1, 0), (404, 1, 0), (917, 1, 0), (406, 1, 0), (919, 1, 0), (408, 1, 0), (409, 1, 0), (410, 1, 0), (416, 1, 0), (929, 1, 0), (420, 1, 0), (935, 1, 0), (937, 1, 0), (426, 1, 0), (939, 1, 0), (428, 1, 0), (941, 1, 0), (430, 1, 0), (943, 1, 0), (431, 1, 0), (945, 1, 0), (434, 1, 0), (443, 1, 0), (444, 1, 0), (963, 1, 0), (452, 1, 0), (965, 1, 0), (967, 1, 0), (456, 1, 0), (457, 1, 0), (458, 1, 0), (460, 1, 0), (462, 1, 0), (464, 1, 0), (979, 1, 0), (468, 1, 0), (983, 1, 0), (473, 1, 0), (474, 1, 0), (475, 1, 0), (476, 1, 0), (477, 1, 0), (478, 1, 0), (479, 1, 0), (993, 1, 0), (484, 1, 0), (489, 1, 0), (491, 1, 0), (1005, 1, 0), (1007, 1, 0), (495, 1, 0), (1009, 1, 0), (498, 1, 0), (1011, 1, 0), (505, 1, 0), (507, 1, 0), (510, 1, 0)]

fifth_phase_dist_data = [(55, 1, 1), (760, 1, 1), (764, 1, 1), (1020, 1, 1), (763, 1, 1), (311, 1, 1), (439, 1, 1)]



fifth_phase_valid_data = [(0, 1, 0), (2, 1, 0), (8, 1, 0), (9, 1, 0), (521, 1, 0), (523, 1, 0), (10, 1, 0), (13, 1, 0), (11, 1, 0), (15, 1, 0), (525, 1, 0), (16, 1, 0), (18, 1, 0), (20, 1, 0), (533, 1, 0), (535, 1, 0), (25, 1, 0), (26, 1, 0), (28, 1, 0), (29, 1, 0), (34, 1, 0), (38, 1, 0), (43, 1, 0), (555, 1, 0), (557, 1, 0), (45, 1, 0), (559, 1, 0), (44, 1, 0), (561, 1, 0), (50, 1, 0), (54, 1, 0), (567, 1, 0), (56, 1, 0), (60, 1, 0), (61, 1, 0), (62, 1, 0), (63, 1, 0), (66, 1, 0), (72, 1, 0), (73, 1, 0), (587, 1, 0), (77, 1, 0), (589, 1, 0), (78, 1, 0), (593, 1, 0), (82, 1, 0), (597, 1, 0), (86, 1, 0), (599, 1, 0), (90, 1, 0), (94, 1, 0), (95, 1, 0), (609, 1, 0), (98, 1, 0), (102, 1, 0), (105, 1, 0), (106, 1, 0), (108, 1, 0), (109, 1, 0), (110, 1, 0), (623, 1, 0), (112, 1, 0), (111, 1, 0), (114, 1, 0), (118, 1, 0), (631, 1, 0), (121, 1, 0), (122, 1, 0), (124, 1, 0), (126, 1, 0), (641, 1, 0), (649, 1, 0), (651, 1, 0), (140, 1, 0), (141, 1, 0), (653, 1, 0), (143, 1, 0), (144, 1, 0), (657, 1, 0), (146, 1, 0), (659, 1, 0), (148, 1, 0), (661, 1, 0), (150, 1, 0), (152, 1, 0), (155, 1, 0), (160, 1, 0), (675, 1, 0), (164, 1, 0), (166, 1, 0), (679, 1, 0), (168, 1, 0), (169, 1, 0), (170, 1, 0), (683, 1, 0), (172, 1, 0), (173, 1, 0), (171, 1, 0), (175, 1, 0), (176, 1, 0), (687, 1, 0), (178, 1, 0), (691, 1, 0), (180, 1, 0), (693, 1, 0), (182, 1, 0), (695, 1, 0), (184, 1, 0), (187, 1, 0), (188, 1, 0), (189, 1, 0), (191, 1, 0), (192, 1, 0), (705, 1, 0), (194, 1, 0), (196, 1, 0), (709, 1, 0), (198, 1, 0), (711, 1, 0), (713, 1, 0), (202, 1, 0), (204, 1, 0), (717, 1, 0), (206, 1, 0), (207, 1, 0), (210, 1, 0), (212, 1, 0), (214, 1, 0), (727, 1, 0), (217, 1, 0), (218, 1, 0), (221, 1, 0), (222, 1, 0), (223, 1, 0), (737, 1, 0), (226, 1, 0), (741, 1, 0), (745, 1, 0), (747, 1, 0), (236, 1, 0), (237, 1, 0), (749, 1, 0), (240, 1, 0), (242, 1, 0), (755, 1, 0), (244, 1, 0), (757, 1, 0), (759, 1, 0), (248, 1, 0), (251, 1, 0), (252, 1, 0), (253, 1, 0), (262, 1, 0), (775, 1, 0), (266, 1, 0), (779, 1, 0), (268, 1, 0), (781, 1, 0), (269, 1, 0), (783, 1, 0), (272, 1, 0), (785, 1, 0), (274, 1, 0), (787, 1, 0), (276, 1, 0), (281, 1, 0), (282, 1, 0), (286, 1, 0), (287, 1, 0), (801, 1, 0), (803, 1, 0), (805, 1, 0), (296, 1, 0), (809, 1, 0), (300, 1, 0), (302, 1, 0), (303, 1, 0), (306, 1, 0), (819, 1, 0), (821, 1, 0), (310, 1, 0), (823, 1, 0), (312, 1, 0), (313, 1, 0), (314, 1, 0), (318, 1, 0), (319, 1, 0), (837, 1, 0), (326, 1, 0), (329, 1, 0), (330, 1, 0), (332, 1, 0), (845, 1, 0), (334, 1, 0), (336, 1, 0), (849, 1, 0), (851, 1, 0), (853, 1, 0), (342, 1, 0), (349, 1, 0), (350, 1, 0), (352, 1, 0), (354, 1, 0), (867, 1, 0), (356, 1, 0), (869, 1, 0), (358, 1, 0), (871, 1, 0), (360, 1, 0), (873, 1, 0), (362, 1, 0), (363, 1, 0), (361, 1, 0), (365, 1, 0), (877, 1, 0), (367, 1, 0), (368, 1, 0), (366, 1, 0), (370, 1, 0), (875, 1, 0), (382, 1, 0), (384, 1, 0), (386, 1, 0), (899, 1, 0), (388, 1, 0), (390, 1, 0), (393, 1, 0), (394, 1, 0), (395, 1, 0), (911, 1, 0), (400, 1, 0), (913, 1, 0), (402, 1, 0), (917, 1, 0), (406, 1, 0), (408, 1, 0), (418, 1, 0), (933, 1, 0), (935, 1, 0), (424, 1, 0), (937, 1, 0), (426, 1, 0), (428, 1, 0), (429, 1, 0), (431, 1, 0), (943, 1, 0), (945, 1, 0), (949, 1, 0), (438, 1, 0), (441, 1, 0), (443, 1, 0), (445, 1, 0), (961, 1, 0), (965, 1, 0), (967, 1, 0), (969, 1, 0), (458, 1, 0), (461, 1, 0), (464, 1, 0), (470, 1, 0), (983, 1, 0), (472, 1, 0), (473, 1, 0), (474, 1, 0), (475, 1, 0), (480, 1, 0), (995, 1, 0), (484, 1, 0), (486, 1, 0), (999, 1, 0), (488, 1, 0), (1001, 1, 0), (490, 1, 0), (489, 1, 0), (1003, 1, 0), (495, 1, 0), (496, 1, 0), (498, 1, 0), (1011, 1, 0), (500, 1, 0), (1013, 1, 0), (504, 1, 0), (506, 1, 0), (508, 1, 0), (509, 1, 0), (510, 1, 0), (511, 1, 0)]
number_of_vars = 10
df_dist = prepare_data(initial_dist_data, number_of_vars, DistEstimate = True)
df_valid = prepare_data(initial_valid_data, number_of_vars, DistEstimate = False)
df = pd.concat([df_dist, df_valid], ignore_index=True)
X = df[list(df.columns[:-3])].values
y = df['label'].values
clf = CustomDecisionTree(max_depth= number_of_vars)
clf.fit_initial(X, y, feature_names=df.columns[:-3])
#print("initial dnf", clf.tree_to_dnf())
clf.save_tree(f'{PATH}/ex7_1')
predictions = clf.predict(df_dist[list(df.columns[:-3])].values, expected_labels=df_dist['label'].values, weights=df_dist['weight'].values, member=df_dist['member'].values)
print("Initial Bounds(before prediction):", clf.get_error_bounds())
clf.save_tree(f'{PATH}/ex7_2')
print("Tree DNF:", clf.tree_to_dnf())


#print("dnf", clf.tree_to_dnf())
df_dist_second = prepare_data(second_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_second = prepare_data(second_phase_valid_data, number_of_vars, DistEstimate = False)
df_second = pd.concat([df_dist_second, df_valid_second], ignore_index=True)
clf.predict(df_second[list(df.columns[:-3])].values, expected_labels=df_second['label'].values, weights=df_second['weight'].values, member=df_second['member'].values)
print("Initial Bounds(after prediction):", clf.get_error_bounds())

df_dist_third = prepare_data(third_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_third = prepare_data(third_phase_valid_data, number_of_vars, DistEstimate = False)
df_third = pd.concat([df_dist_third, df_valid_third], ignore_index=True)

df_dist_fourth = prepare_data(fourth_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_fourth = prepare_data(fourth_phase_valid_data, number_of_vars, DistEstimate = False)
df_fourth = pd.concat([df_dist_fourth, df_valid_fourth], ignore_index=True)


df_dist_fifth = prepare_data(fifth_phase_dist_data, number_of_vars, DistEstimate = True)
df_valid_fifth = prepare_data(fifth_phase_valid_data, number_of_vars, DistEstimate = False)
df_fifth = pd.concat([df_dist_fifth, df_valid_fifth], ignore_index=True)




for iteration in range(5):
    clf_copy = copy.deepcopy(clf)
    print(f'{iteration=}')
    random.seed(iteration + 10)
    print("Second Phase")
    final_tree, tree_sequence = MuteTree(clf_copy, df_second, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex7_second_{iteration}')
    
    print("Tree DNF:", final_tree.tree_to_dnf())
    
    print("Third Phase")
    final_tree, tree_sequence = MuteTree(final_tree, df_third, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex7_third_{iteration}')
    #print("DNF:", final_tree.tree_to_dnf())
    print("Tree DNF:", final_tree.tree_to_dnf())
    print("Fourth Phase")
    final_tree, tree_sequence = MuteTree(final_tree, df_fourth, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex7_fourth_{iteration}')
    print("Tree DNF:", final_tree.tree_to_dnf())
    

    print("Fifth Phase")
    final_tree, tree_sequence = MuteTree(final_tree, df_fifth, 0, 0)
    final_tree.save_tree(output_file=f'{PATH}/final_trees/ex7_fifth_{iteration}')
    print("Tree DNF:", final_tree.tree_to_dnf())
    



    
    








