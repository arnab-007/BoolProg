import re

# Helper functions to parse and transform formulas
def parse_dnf(dnf_formula):
    """
    Parse the DNF string into a list of conjunctions (terms), where each conjunction is a list of literals.
    """
    # Remove any white spaces and standardize operators
    dnf_formula = dnf_formula.replace("&&", "&").replace("||", "|").replace("!", "~")
    # Split the DNF formula by the OR operator
    terms = dnf_formula.split("|")
    conjunctions = []

    for term in terms:
        literals = term.strip("() ").split("&")
        conjunctions.append([literal.strip() for literal in literals])

    return conjunctions

def tseitin_clause_for_conjunction(conjunction, auxiliary_variable):
    """
    Convert a conjunction to its CNF form using Tseitin transformation and an auxiliary variable.
    For a conjunction of literals like (x1 && x2), it creates the CNF:
    (auxiliary_variable OR NOT x1 OR NOT x2) AND (NOT auxiliary_variable OR x1) AND (NOT auxiliary_variable OR x2)
    """
    cnf_clauses = []
    # Create the clause representing the conjunction
    neg_conjunction = [f"~{literal}" for literal in conjunction]

    # Clause for the auxiliary variable representing the conjunction
    cnf_clauses.append(f"({auxiliary_variable} | {' | '.join(neg_conjunction)})")

    # Clauses for the negation of the auxiliary variable
    for literal in conjunction:
        cnf_clauses.append(f"(~{auxiliary_variable} | {literal})")

    return cnf_clauses

def remove_double_negations(cnf_formula):
    """
    Remove double negations (~~x -> x) from the CNF formula.
    """
    # Use a regular expression to remove double negations
    cnf_formula = re.sub(r'~~([a-zA-Z0-9_]+)', r'\1', cnf_formula)
    return cnf_formula

def dnf_to_cnf_tseitin(dnf_formula):
    """
    Convert the given DNF formula (in string format) to an equisatisfiable CNF formula using Tseitin's transformation.
    """
    # Parse the DNF formula into conjunctions
    conjunctions = parse_dnf(dnf_formula)

    auxiliary_variables = []
    cnf_clauses = []
    var_counter = 1

    # For each conjunction, introduce a new auxiliary variable and convert it to CNF
    for conjunction in conjunctions:
        auxiliary_variable = f"a{var_counter}"
        auxiliary_variables.append(auxiliary_variable)

        # Get the CNF clauses for this conjunction
        clauses_for_conjunction = tseitin_clause_for_conjunction(conjunction, auxiliary_variable)

        # Add the clauses to the CNF formula
        cnf_clauses.extend(clauses_for_conjunction)
        
        # Increment the auxiliary variable counter
        var_counter += 1

    # Add the final clause to link the overall DNF formula to the auxiliary variables
    dnf_clause = " | ".join(auxiliary_variables)
    cnf_clauses.append(f"({dnf_clause})")

    # Combine all clauses into a CNF formula
    cnf_formula = " & ".join([f"({clause})" for clause in cnf_clauses])

    # Remove double negations
    cnf_formula = remove_double_negations(cnf_formula)

    return cnf_formula,auxiliary_variables

# Example usage
dnf_formula = "(!x2 && !x7 && x5) || (x7 && x5 && !x1) || (x8 && !x9) || (x1 && x4 && !x6 && !x8) || (x2 && !x3 && x4) || (!x3 && !x4 && x5 && x6)"
cnf_formula ,auxiliary_variables = dnf_to_cnf_tseitin(dnf_formula)
print("DNF Formula:", dnf_formula)
print("Auxiliary variables:",auxiliary_variables)
print("Equisatisfiable CNF Formula:", cnf_formula.replace('~','!').replace('|','||').replace('&','&&'))


