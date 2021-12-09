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

# Defining helpher functions to substitute numerical values into the SymPy formulas
vars = (x, y, x_0, y_0, alpha, alpha_0)
sub_v_x = sympy.lambdify(vars, v_x, modules="numpy")
sub_v_y = sympy.lambdify(vars, v_y, modules="numpy")
sub_alpha = sympy.lambdify(vars, 0.5 * curl, modules="numpy")

def substitute_position_values(x, y, alpha, x_0, y_0, alpha_0):
    """Substitutes the position coordinates (x, y, alpha) of one particle and (x_0, y_0, alpha_0)
       of another particle into the expressions for v_x, v_y, and alpha.
       
       This is used in the modelling function to calculate the total forces and torques on
       the object."""
    return (sub_v_x(x, y, alpha, x_0, y_0, alpha_0), sub_v_y(x, y, alpha, x_0, y_0, alpha_0), \
            sub_alpha(x, y, alpha, x_0, y_0, alpha_0))

# def model(i):
#     # Using library functions as much as possible to reduce overhead and improve performance
#     # This function uses a "blessed index" approach to perform the calculations while excluding
#     # a single index of the ndarray.
#     return np.sum(substitute_position_values(position[i], position[:i])) + \
#            np.sum(substitute_position_values(position[i], position[i + 1:]))


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