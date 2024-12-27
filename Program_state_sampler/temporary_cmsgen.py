import re

def parse_cnf_to_dimacs(cnf_formula, output_file):
    # Step 1: Preprocess CNF formula
    cnf_formula = cnf_formula.replace("&&", "and").replace("||", "or")
    cnf_formula = cnf_formula.replace("(", "").replace(")", "")
    clauses = cnf_formula.split("and")  # Split into individual clauses

    # Step 2: Map variables to unique integers
    variables = set()
    parsed_clauses = []

    for clause in clauses:
        literals = clause.split("or")
        literals = [lit.strip() for lit in literals]  # Remove whitespace
        parsed_clauses.append(literals)
        for lit in literals:
            # Extract variable name (strip negations)
            var = lit.lstrip("!")
            variables.add(var)

    variables = sorted(variables)  # Sort variables for consistency
    var_to_int = {var: i + 1 for i, var in enumerate(variables)}

    # Step 3: Convert clauses to integer format
    dimacs_clauses = []
    for clause in parsed_clauses:
        dimacs_clause = []
        for lit in clause:
            if lit.startswith("!"):
                dimacs_clause.append(-var_to_int[lit[1:]])  # Negated variable
            else:
                dimacs_clause.append(var_to_int[lit])  # Positive variable
        dimacs_clauses.append(dimacs_clause)

    # Step 4: Write to DIMACS file
    with open(output_file, "w") as f:
        f.write(f"p cnf {len(variables)} {len(dimacs_clauses)}\n")
        for dimacs_clause in dimacs_clauses:
            f.write(" ".join(map(str, dimacs_clause)) + " 0\n")

# Example CNF formula
cnf_formula = """
((a1 || x2 || x7 || !x5)) && ((!a1 || !x2)) && ((!a1 || !x7)) && ((!a1 || x5)) &&
((a2 || !x7 || !x5 || x1)) && ((!a2 || x7)) && ((!a2 || x5)) && ((!a2 || !x1)) &&
((a3 || !x8 || x9)) && ((!a3 || x8)) && ((!a3 || !x9)) &&
((a4 || !x1 || !x4 || x6 || x8)) && ((!a4 || x1)) && ((!a4 || x4)) &&
((!a4 || !x6)) && ((!a4 || !x8)) &&
((a5 || !x2 || x3 || !x4)) && ((!a5 || x2)) && ((!a5 || !x3)) && ((!a5 || x4)) &&
((a6 || x3 || x4 || !x5 || !x6)) && ((!a6 || !x3)) && ((!a6 || !x4)) && ((!a6 || x5)) &&
((!a6 || x6)) && ((a1 || a2 || a3 || a4 || a5 || a6))
"""

# Output file
output_file = "formula.cnf"

# Parse and write to DIMACS
parse_cnf_to_dimacs(cnf_formula, output_file)

print(f"CNF formula written to {output_file}")



X = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
T = ["a1", "a2", "a3", "a4", "a5", "a6"]

#samples = sample_x_using_cmsgen(F_str, X, T)
#print("Sampled X assignments:", samples)

