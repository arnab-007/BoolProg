def generate_DIMACS_formula(formula, variable_mapping):
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
    
    num_variables = len(variable_mapping)
    num_clauses = len(dimacs_formula)
    print(dimacs_formula)
    dimacs_str = f"p cnf {num_variables} {num_clauses}\n"
    print(dimacs_str)
    
    for clause in dimacs_formula:
        dimacs_str += ' '.join(str(i) for i in clause) + ' 0\n'
    
    return dimacs_str