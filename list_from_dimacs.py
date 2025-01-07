def extract_formula_from_DIMACS(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('p cnf'):
                parts = line.strip().split()
                num_variables = int(parts[2])
                num_clauses = int(parts[3])
                formula = []
                for _ in range(num_clauses):
                    clause = [int(x) for x in next(file).split() if x != '0']
                    formula.append(clause)


                return [formula ,num_variables]

                

