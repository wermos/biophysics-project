from common import *
from scipy.integrate import solve_ivp

x, y, x_0, y_0, alpha, alpha_0 = sympy.symbols("x y x_0 y_0 alpha alpha_0")
"""
x       = x-coordinate of a dipole
y       = y-coordinate of a dipole
alpha   = angle of that dipole w.r.t. the positive x-axis
x_0     = x-coordinate of the dipole w.r.t. we are doing our calculations
y_0     = y-coordinate of the dipole w.r.t. we are doing our calculations
alpha_0 = angle (w.r.t. the positive x-axis) of the dipole w.r.t. we are doing our calculations
"""

leading_constant = -kappa / (4 * sympy.pi * eta_2D)
subexp_1 = (x - x_0) / r(x, y, x_0, y_0)
subexp_2 = (y - y_0) / r(x, y, x_0, y_0)

subexp_3 = ( subexp_1 * sympy.cos(alpha_0) ) + ( subexp_2 * sympy.sin(alpha_0) )

subexp_4 = 1 - 2 * (subexp_3 ** 2)

v_x = (leading_constant / r(x, y, x_0, y_0)) * subexp_4 * subexp_1
v_y = (leading_constant / r(x, y, x_0, y_0)) * subexp_4 * subexp_2

curl = sympy.diff(v_y, x) - sympy.diff(v_x, y)

def substitute_position_values(coords_i, coords_j):
    """Substitutes the position coordinates (x, y, alpha) of particles i and j into the
       expressions for v_x, v_y, and alpha.
       
       This is used in the modelling function `model` to calculate the total forces and
       torques on the object."""
    sub_v_x = sympy.N(v_x, \
                {x: coords_j[0], y: coords_j[1], alpha: coords_j[2], \
                x_0: coords_i[0], y_0: coords_i[1], alpha_0: coords_i[2]})
    sub_v_y = sympy.N(v_y, \
                {x: coords_j[0], y: coords_j[1], alpha: coords_j[2], \
                x_0: coords_i[0], y_0: coords_i[1], alpha_0: coords_i[2]})
    sub_alpha = 0.5 * sympy.N(curl, \
                {x: coords_j[0], y: coords_j[1], alpha: coords_j[2], \
                x_0: coords_i[0], y_0: coords_i[1], alpha_0: coords_i[2]})
    return (sub_v_x, sub_v_y, sub_alpha)


# def total_y_sum(i):
#     """Finds the total force in the y-direction for the i'th particle."""
#     # temp = np.sum(position[:i], axis=0) + np.sum(position[i + 1:], axis=0)
#     # temp[2] /= 2
#     # return temp
#     total = 0
#     for j in range(num_dipoles):
#         if i != j:
#             total += 

def model(i):
    # Using library functions as much as possible to reduce overhead and improve performance
    # This function uses a "blessed index" approach to perform the calculations while excluding
    # a single index of the ndarray.
    return np.sum(substitute_position_values(position[i], position[:i])) + \
           np.sum(substitute_position_values(position[i], position[i + 1:]))

# Initial positions of both particles
# Particle 1
x_1 = y_1 = alpha_1 = 0
# Particle 2
x_2 = 0.4
y_2 = 0
alpha_2 = sympy.N(sympy.pi / 6)

position[0][0] = x_1
position[0][1] = y_1
position[0][2] = alpha_1

position[1][0] = x_2
position[1][1] = y_2
position[1][2] = alpha_2

time = np.linspace(0, 200, N)

data = solve_ivp(model, (0, 200), position, t_eval=time)

# plt.plot(time, data.x)