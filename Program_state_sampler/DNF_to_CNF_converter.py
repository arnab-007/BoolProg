from sympy.logic.boolalg import Or, And, Not, simplify_logic
from sympy import symbols
import re

def dnf_to_cnf_no_aux(dnf_formula):
    """
    Convert a DNF formula into an equivalent CNF formula without auxiliary Tseitin variables.
    Args:
        dnf_formula (str): DNF formula as a string, e.g., "(x1 && x2) || (x3 && x4)"
    Returns:
        str: Equivalent CNF formula as a string only using the original variables.
    """
    # Step 1: Preprocess the formula
    dnf_formula = dnf_formula.replace("&&", "&").replace("||", "|").replace("!", "~")
    
    # Step 2: Extract unique variables using regex
    variables = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', dnf_formula))
    print("Variables:",variables)
    sympy_vars = {var: symbols(var) for var in variables}
    print("Sympy variables:",sympy_vars)
    # Step 3: Parse the formula into a sympy expression
    try:
        dnf_expr = eval(dnf_formula, {"__builtins__": None, **sympy_vars, "&": And, "|": Or, "~": Not})
    except Exception as e:
        raise ValueError(f"Failed to parse DNF formula. Error: {e}")
    print("DNF expression:",dnf_expr)
    # Step 4: Use distributive property to convert DNF to CNF directly
    cnf_expr = simplify_logic(dnf_expr, form='cnf')
    print("CNF expression:",cnf_expr)
    # Step 5: Convert CNF expression to string format
    cnf_string = str(cnf_expr).replace("&", "&&").replace("|", "||").replace("~", "!")
    return cnf_string


# Example Usage
if __name__ == "__main__":
    #dnf_formula = "(x1 && !x2 && x4 && !x5) || (x2 && !x3 && x4 && x5) || (x2 && !x3 && !x4 && x7) || (x4 && !x5 && x6 && !x2) && (x3 && !x4 && x5 && x7) || (x1 && x2 && !x3 && !x5) || (x3 && !x4 && x5 && !x6) || (x3 && !x4 && !x5 && !x6) && (!x1 && x3 && !x4 && x5)"
    dnf_formula_first = "(x1 && x2 && x3) || (x1 && !x2 && x3) || (x1 && x3) || (!x1 && x2 && x3)"
    #dnf_formula = "(x1 && !x2 && x4 && !x5) || (x2 && !x3 && x4 && x5) || (x2 && !x3 && !x4 && x7) || (x4 && !x5 && x6 && !x2) && (x3 && !x4 && x5 && x7) || (x1 && x2 && !x3 && !x5) || (x3 && !x4 && x5 && !x6) || (x3 && !x4 && !x5 && !x6) && (!x1 && x3 && !x4 && x5)"
    #dnf_formula = "(!x4 && !x10 && !x8 && !x2 && !x9 && !x1 && !x5) || (!x4 && !x10 && !x8 && !x2 && !x9 && x1 && x5) || (!x4 && !x10 && !x8 && !x2 && x9 && !x11 && !x3 && !x7) || (!x4 && !x10 && !x8 && !x2 && x9 && !x11 && x3) || (!x4 && !x10 && !x8 && !x2 && x9 && x11) || (!x4 && !x10 && !x8 && x2 && !x12 && x3 && !x6 && !x7) || (!x4 && !x10 && !x8 && x2 && !x12 && x3 && !x6 && x7 && x9) || (!x4 && !x10 && !x8 && x2 && !x12 && x3 && x6 && x11 && !x1) || (!x4 && !x10 && !x8 && x2 && !x12 && x3 && x6 && x11 && x1 && x5) || (!x4 && !x10 && !x8 && x2 && x12 && !x1 && !x9) || (!x4 && !x10 && !x8 && x2 && x12 && x1) || (!x4 && !x10 && x8 && !x9) || (!x4 && !x10 && x8 && x9 && !x5 && !x1) || (!x4 && !x10 && x8 && x9 && x5 && x1) || (x4 && x10 && !x6 && !x5 && !x1) || (x4 && x10 && !x6 && x5 && x1) || (x4 && x10 && x6 && !x8 && !x11 && x2) || (x4 && x10 && x6 && !x8 && x11 && !x2) || (x4 && x10 && x6 && !x8 && x11 && x2 && !x5) || (x4 && x10 && x6 && x8 && !x9) || (x4 && x10 && x6 && x8 && x9 && !x11) || (x4 && x10 && x6 && x8 && x9 && x11 && x1)"
    dnf_formula = "(!x2 && !x7 && x5) || (x7 && x5 && !x1) || (!x8 && !x9 && x2 && x9 && !x6) || (x8 && !x9) || (x2) || (!x7 && x4 && !x1) || (!x6) || (!x3 && x1 && x6) || (!x1 && !x5 && !x3 && x9) || (!x4 && x7 && !x5 && x9 && !x1) || (!x3 && !x2) || (x4 && x6 && x7) || (!x8 && x5 && !x7 && !x7) || (x5) || (x6) || (!x10 && x5 && !x4 && !x10 && !x4) || (!x8 && !x10 && x6) || (!x1) || (!x8 && x8 && x8 && !x5 && !x1) || (x6 && !x3)"
    dnf_formula = "(x1 && !x2) || (!x1 && x2)"
    cnf_formula = dnf_to_cnf_no_aux(dnf_formula)
    print("Input DNF Formula:", dnf_formula)
    print("Output CNF Formula:", cnf_formula)


#\




dnf_formula = "(x1 && !x2 && x4 && !x5) || (x2 && !x3 && x4 && x5) || (x2 && !x3 && !x4 && x7) || (x4 && !x5 && x6 && !x2) && (x3 && !x4 && x5 && x7) || (x1 && x2 && !x3 && !x5) || (x3 && !x4 && x5 && !x6) || (x3 && !x4 && !x5 && !x6) && (!x1 && x3 && !x4 && x5)"
dnf_formula = "(x1 && x2 && x4 && x5)"


