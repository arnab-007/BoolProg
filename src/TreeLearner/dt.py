import z3
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

def block_model(s):
    m = s.model()
    s.add(z3.Or([f() != m[f] for f in m.decls() if f.arity() == 0]))


def get_solutions(s, no_sols):
    solutions = []
    for _ in range(no_sols):
        status = s.check()
        if status == z3.sat:
            solutions.append(s.model())
            block_model(s)
    return solutions

def get_solutions_complex(s, no_sols, var_list):
    solutions = []
    for _ in range(no_sols):
        status = s.check()
        if status == z3.sat:
            solutions.append(
                {d: s.model().eval(d, model_completion=True) for d in var_list}
            )
            block_model(s)
    return solutions

# Declare variables
a0, a1, a2, b0, b1, b2, b3, c0, c1, d0, d1, e0, e1 = z3.Bools('a0 a1 a2 b0 b1 b2 b3 c0 c1 d0 d1 e0 e1')

# Define the CNF fz3.Ormula
prog_cnf = z3.And(
    z3.Or(a1, a0), z3.Or(a1, z3.Not(d0)), z3.Or(z3.Not(a1), z3.Not(a0), d0),
    z3.Or(z3.Not(b1), b0), z3.Or(z3.Not(b1), a1), z3.Or(z3.Not(b1), e0),
    z3.Or(z3.Not(b2), z3.Not(b1), a1), z3.Or(z3.Not(b2), c0),
    z3.Or(z3.Not(c1), z3.Not(c0)), z3.Or(z3.Not(c1), e0),
    z3.Or(z3.Not(d1), c1), z3.Or(z3.Not(d1), b2), z3.Or(z3.Not(d1), e0),
    z3.Or(z3.Not(a2), b2), z3.Or(z3.Not(a2), c1), z3.Or(z3.Not(a2), e0),
    z3.Or(z3.Not(b3), c1), z3.Or(z3.Not(b3), b2, e0), z3.Or(z3.Not(b3), z3.Not(a2), e0),
    z3.Or(z3.Not(e1), d1), z3.Or(z3.Not(e1), a2)
)

'''gpt suggested candidate = z3.And(z3.Implies(e0, z3.Not(a0)), c0 == e0, z3.Implies(d0, z3.And(e0, z3.Not(b0))))
but this is also unsat = z3.And(c0 == e0, z3.Implies(d0, z3.And(e0, z3.Not(b0)))),
Finally this is invaraint: c == e
'''
inv_pre = z3.Or(c0, e0)
inv_post = z3.Implies(c1, e1)

true_samples = []
false_samples = []

# inv_pre = z3.And(z3.Or(z3.Not(a0), b0), z3.Or(z3.Not(a0), d0), z3.Or(z3.Not(c0), z3.Not(d0)),
# z3.Or(d0, z3.Not(e0)), z3.Or(a0, z3.Not(b0), d0))

# inv_post = z3.Not(z3.And(z3.Or(z3.Not(a2), b3), z3.Or(z3.Not(a2), d1), z3.Or(z3.Not(c1), z3.Not(d1)),
# z3.Or(d1, z3.Not(e1)), z3.Or(a2, z3.Not(b3), d1)))

# prog_cnf = z3.And(inv_pre, prog_cnf, inv_post)

# Optionally, create a solver z3.z3.And add the fz3.z3.Ormula to it
s = z3.Solver()

s.add(z3.And(inv_pre, prog_cnf, inv_post))
true_samples = get_solutions_complex(s, 11, [a0, a1, a2, b0, b1, b2, b3, c0, c1, d0, d1, e0, e1])
s.reset()
s.add(z3.And(inv_pre, prog_cnf, z3.Not(inv_post)))
false_samples = get_solutions_complex(s, 20, [a0, a1, a2, b0, b1, b2, b3, c0, c1, d0, d1, e0, e1])




true_samples = [{str(k):v for k,v in ele.items() if k in [a0,b0,c0,d0,e0]} for ele in true_samples]
false_samples = [{str(k):v for k,v in ele.items() if k in [a0,b0,c0,d0,e0]} for ele in false_samples]

ts = pd.DataFrame(true_samples).drop_duplicates()
ts['label'] = 1
fs = pd.DataFrame(false_samples).drop_duplicates()
fs['label'] = 0


print(ts)

print(fs)
# print(len(ts))
# print(len(fs))


final_df = pd.concat([ts, fs], axis=0, ignore_index=True)

def convert_boolref_to_int(value):
    # Convert z3.BoolRef to int
    return int(bool(z3.BoolVal(value)))
# Apply the conversion to each cell in the DataFrame
for column in final_df.columns[:-1]:  # Exclude the 'label' column
    final_df[column] = final_df[column].apply(convert_boolref_to_int)


X = final_df.drop('label', axis=1)
Y = final_df['label']

# Fit the decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=12, min_impurity_decrease=0.1, random_state=0)
clf = clf.fit(X, Y)



plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['0', '1'], rounded=True, fontsize=14)
plt.show()

# # Get the leaf index for each sample
# leaf_indices = clf.apply(X)
# final_df['leaf'] = leaf_indices

# # Display which samples went to which leaf
# print(final_df[['label', 'leaf']])

# # Inspect the depth and number of leaves
# print(f"Tree depth: {clf.get_depth()}")
# print(f"Number of leaves: {clf.get_n_leaves()}")

# s.add(z3.And(inv_pre, prog_cnf, inv_post))
# true_samples = get_solutions(s, 20)
# s.reset()
# s.add(z3.And(inv_pre, prog_cnf, z3.Not(inv_post)))
# false_samples = get_solutions(s, 20)
# print(true_samples)
# print(false_samples)
# print(len(true_samples))
# print(len(false_samples))


# def combine_df(ts, fs):
#     ts = [{d.name(): ele[d] for d in ele.decls() if d.name() in ['a0','b0','c0','d0','e0']} for ele in ts]
#     fs = [{d.name(): ele[d] for d in ele.decls() if d.name() in ['a0','b0','c0','d0','e0']} for ele in fs]
#     print(ts)
#     print(fs)
#     # ts = pd.DataFrame(ts)
#     # fs = pd.DataFrame(fs)
#     # frames = [ts, fs]
#     # final_df = pd.concat(frames, axis=0, ignore_index=True)
#     # return final_df


# print(combine_df(true_samples, false_samples))
# Print the fz3.z3.Ormula to verify
# print(prog_cnf)

# # Check satisfiability
# if solver.check() == z3.sat:
#     print("The fz3.z3.Ormula is satisfiable.")
#     model = solver.model()
#     print(model)
# else:
#     print("The fz3.z3.Ormula is unsatisfiable.")

# # This `cnf_fz3.Ormula` variable now contains the entire CNF fz3.Ormula encoded in Z3




# # Define z3.Boolean variables
# a1 = z3.Bool('a1')
# a2 = z3.Bool('a2')
# b1 = z3.Bool('b1')
# b2 = z3.Bool('b2')

# # Define the CNF clauses
# clauses = [
#     z3.z3.Or(z3.z3.Not(a1), a2),            # (¬a1 ∨ a2)
#     z3.z3.Or(z3.z3.Not(b1), a2),            # (¬b1 ∨ a2)
#     z3.z3.Or(z3.z3.Not(a2), a1, b1),        # (¬a2 ∨ a1 ∨ b1)
#     z3.z3.Or(a2, z3.z3.Not(b2)),            # (a2 ∨ ¬b2)
#     z3.z3.Or(b1, z3.z3.Not(b2)),            # (b1 ∨ ¬b2)
#     z3.z3.Or(b2, z3.z3.Not(a2), z3.z3.Not(b1))    # (b2 ∨ ¬a2 ∨ ¬b1)
# ]

# # inv_pre =  [z3.z3.And(z3.z3.Or(a1, z3.z3.Not(b1)), z3.z3.Not(a1))]
# # inv_post = [z3.z3.Not(z3.z3.And(z3.z3.Or(a2, z3.z3.Not(b2)), z3.z3.Not(a2)))]

# inv_pre =  [b1]
# inv_post = [z3.z3.Not(b2)]


# clauses = inv_pre + clauses + inv_post
# # Combine clauses into a single fz3.z3.Ormula
# prog_cnf = z3.z3.And(clauses)

# # Optionally, create a solver z3.z3.And add the fz3.z3.Ormula to it
# solver = z3.Solver()
# solver.add(prog_cnf)

# # Print the fz3.z3.Ormula to verify
# # print(prog_cnf)

# # Check satisfiability
# if solver.check() == z3.sat:
#     print("The fz3.z3.Ormula is satisfiable.")
#     model = solver.model()
#     print(model)
# else:
#     print("The fz3.z3.Ormula is unsatisfiable.")
