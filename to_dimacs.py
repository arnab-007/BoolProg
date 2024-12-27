<<<<<<< HEAD
def generate_init_DIMACS_formula(formula, variable_mapping,min_indices):

    dimacs_formula = []
    clauses = formula.split('&&')

    for clause in clauses:
        literals = clause.split('||')
        dimacs_clause = []
        
        # Remove extra characters and strip whitespace
        literals = [literal.strip("() ") for literal in literals]
        literals = list(filter(None, literals))  # Filter out empty strings
        #print("Literals in clause:", literals)  # Debugging output
        
        for literal in literals:
            # Extract variable name and check for negation
            var_name = literal[1:] if literal.startswith('!') else literal
            
            # Build the key to match `variable_mapping`
            key = var_name + '_' + str(1)
            
            if key in variable_mapping:  # Check if key exists in `variable_mapping`
                if literal.startswith('!'):
                    dimacs_clause.append(-variable_mapping[key])  # Negation
                else:
                    dimacs_clause.append(variable_mapping[key])  # Positive literal
            else:
                print(f"Warning: {key} not found in variable_mapping")  # Key not found
                print(dimacs_clause)
        dimacs_formula.append(dimacs_clause)  # Add the clause to the formula
    #print(dimacs_formula)
=======
def generate_init_DIMACS_formula(formula, variable_mapping):
    dimacs_formula = []
    clauses = formula.split('&&')
    
    for clause in clauses:
        literals = clause.split('||')
        
        dimacs_clause = []
        literals = [literal.strip("() ") for literal in literals]
        # Filter out empty strings
        literals = list(filter(None, literals))
        
        
        for literal in literals:
            
            if literal.startswith('!'):
                dimacs_clause.append(-variable_mapping[literal[1:]+'_1'])
            else:
                dimacs_clause.append(variable_mapping[literal+'_1'])
        
        dimacs_formula.append(dimacs_clause)
>>>>>>> d24bd43a4e465d7cbddd210d89b87469967e9ddf
    
    num_variables = len(variable_mapping)
    num_clauses = len(dimacs_formula)
    
    dimacs_str = f"p cnf {num_variables} {num_clauses}\n"
    
    
    for clause in dimacs_formula:
        dimacs_str += ' '.join(str(i) for i in clause) + ' 0\n'
    
    return dimacs_formula , dimacs_str

def generate_final_DIMACS_formula(formula, variable_mapping,updates):
    dimacs_formula = []
    clauses = formula.split('&&')
    
    for clause in clauses:
        literals = clause.split('||')
        
        dimacs_clause = []
        literals = [literal.strip("() ") for literal in literals]
<<<<<<< HEAD

=======
>>>>>>> d24bd43a4e465d7cbddd210d89b87469967e9ddf
        # Filter out empty strings
        literals = list(filter(None, literals))
        
        
        for literal in literals:
            
            if literal.startswith('!'):
<<<<<<< HEAD
                dimacs_clause.append(-variable_mapping[literal[1:]+'_'+str(updates[literal[1:]])])
            else:
                dimacs_clause.append(variable_mapping[literal+'_'+str(updates[literal])])
=======
                dimacs_clause.append(-variable_mapping[literal[1:]+'_'+str(1+updates[literal[1:]])])
            else:
                dimacs_clause.append(variable_mapping[literal+'_'+str(1+updates[literal])])
>>>>>>> d24bd43a4e465d7cbddd210d89b87469967e9ddf
        
        dimacs_formula.append(dimacs_clause)
    
    num_variables = len(variable_mapping)
    num_clauses = len(dimacs_formula)
    
    dimacs_str = f"p cnf {num_variables} {num_clauses}\n"
    
    
    for clause in dimacs_formula:
        dimacs_str += ' '.join(str(i) for i in clause) + ' 0\n'
    
    return dimacs_formula , dimacs_str