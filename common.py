import numpy as np
import sympy


# Hardcoded constants
N = 10_000 # number of steps
eta_2D = 1
kappa = 1

def r(x_1, y_1, x_2, y_2):
    """Returns the Cartesian distance between (x_1, y_1) and (x_2, y_2)."""
    return sympy.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

num_dipoles = 2 # Change as needed

position = np.ndarray([num_dipoles, 3], dtype=np.float64)
"""`position` stores the (x, y, alpha) values of each protein in a 2D array."""
