import numpy as onp
import NURBS_helper_functions as NURBS 


# p = 2  # quadratic
# P = onp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]).reshape(-1)  # 5 control points, 1D for simplicity
# U = NURBS.get_open_uniform_knot_vector(len(P), p)  # get initial knot vector

# print("Original knot vector:", U)
# print("Original control points:\n", P)

# u_bar = 0.5  # insert knot at 0.5

# U_new, P_new = NURBS.knot_insertion(U, P, p, u_bar)

# print("New knot vector:", U_new)
# print("New control points:\n", P_new)


# Original data
p = 2
Pw = onp.array([[0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0]])
knot = NURBS.get_open_uniform_knot_vector(len(Pw), p)

# Insert knot u=0.5
new_knot, new_Pw = NURBS.knot_insertion(knot, p, Pw, 0.5)

print("Original knot vector:", knot)
print("New knot vector:", new_knot)
print("New control points:\n", new_Pw)
