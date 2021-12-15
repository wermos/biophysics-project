from vec2 import *
from particle import *

# Input here the initial conditions of the particles
################################################################################################################################
num_particles = 2

# Particle list
particle_list = []

# Create the particles
particle_list.append(Particle(initial_position=Vec2(0, 0), initial_angle=0))
particle_list.append(Particle(initial_position=Vec2(0.4, 0), initial_angle=np.pi/6))

# Hardcoded constants
## Physical Constants
eta_2D = 1
kappa = 1
## Simulation time
t_start = 0.0
t_end = 200.0
## Number of steps
steps = 10_000
################################################################################################################################

# Helper symbolic expressions
################################################################################################################################
def r(x_a, y_a, x_b, y_b):
	"""Returns the Cartesian distance between (x_a, y_a) and (x_b, y_b)."""
	return sp.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

leading_constant = -kappa / (4 * sp.pi * eta_2D)

subexp_1 = (x_i - x_j) / r(x_i, y_i, x_j, y_j)
subexp_2 = (y_i - y_j) / r(x_i, y_i, x_j, y_j)

subexp_3 = ( subexp_1 * sp.cos(alpha_j) ) + ( subexp_2 * sp.sin(alpha_j) )

subexp_4 = 1 - 2 * (subexp_3 ** 2)

v_x = (leading_constant / r(x_i, y_i, x_j, y_j)) * subexp_4 * subexp_1
v_y = (leading_constant / r(x_i, y_i, x_j, y_j)) * subexp_4 * subexp_2

velocity_vector = Vec2(v_x, v_y)

# curl = 0.5 * (sp.diff(v_y, x_i) - sp.diff(v_x, y_i))
curl = (-1/(4 * sp.pi * eta_2D)) * (sp.cos(alpha_j) * sp.sin(alpha_j)) * (1/(x_i - x_j) ** 2)
# print("curl:")
# print(curl)

# Saving a little memory because why not
del leading_constant, subexp_1, subexp_2, subexp_3, subexp_4
del v_x, v_y
################################################################################################################################

# Create the functions to integrate
for i in range(num_particles):
	particle_list[i].calculate_plane_velocity(particle_list, velocity_vector)
	particle_list[i].calculate_plane_angular_velocity(particle_list, curl)

for i in range(num_particles):
	particle_list[i].lambdify_position()
	particle_list[i].lambdify_velocity(particle_list)
	particle_list[i].lambdify_alpha(particle_list)

# Saving some more memory because we are done with variable and won't need it again
del Particle.var

import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt

# Define the model function
################################################################################################################################
def vector_field(t, old_state):
	"""
	Integrate function.

	The function calculates f, a list with all differential equations of motion in the order
	diff(x0), diff(y0), diff(x1), diff(y1), ...diff(xn-1), diff(yn-1), diff(vx0), diff(vy0)...diff(vxn-1), diff(vyn-1)

	it can be optimized, but it's done to be readable
	"""
	state = np.empty(3 * num_particles, dtype=np.float64)

	for i in range(num_particles):
		state[2 * i] = particle_list[i].lambda_velocity.x(old_state)
		state[2 * i + 1] = particle_list[i].lambda_velocity.y(old_state)
		state[2 * num_particles + i] = particle_list[i].lambda_alpha(old_state)
	return state
################################################################################################################################

# Set the initial conditions
initial_conditions = np.empty(3 * num_particles, dtype=np.float64)
"""For the first 2n elements, it stores (x_i, y_i), and then for the last n elements, it stores alpha_i (1 <= i <= n)."""
for i in range(num_particles):
	initial_conditions[2 * i] = particle_list[i].initial_position.x
	initial_conditions[2 * i + 1] = particle_list[i].initial_position.y
	initial_conditions[2 * num_particles + i] = particle_list[i].initial_angle

# ODE solver parameters
time = np.linspace(t_start, t_end, num=steps)

print("Entering solve_ivp")
sol = solve_ivp(vector_field, (t_start, t_end), initial_conditions, t_eval=time, rtol=1e-5)
# print(sol)
print("Solved the ivp")

fig = plt.figure(figsize=(16, 9))

plt.subplot(1, 3, 1)
plt.plot(sol.t, sol.y[0], color='r', label='particle 1')
plt.plot(sol.t, sol.y[2], color='g', label='particle 2')

plt.xlabel('Time')
plt.ylabel('x coordinate')
plt.legend(['particle 1','particle 2'])

plt.subplot(1, 3, 2)
plt.plot(sol.t, sol.y[1], color='r', label='particle 1')
plt.plot(sol.t, sol.y[3], color='g', label='particle 2')

plt.xlabel('Time')
plt.ylabel('y coordinate')
plt.legend(['particle 1','particle 2'])

plt.subplot(1, 3, 3)
plt.plot(sol.t, sol.y[4], color='r', label='particle 1')
plt.plot(sol.t, sol.y[5], color='g', label='particle 2')

plt.xlabel('Time')
plt.ylabel('Alpha')
plt.legend(['particle 1','particle 2'])
# plt.show()
fig.savefig("temp.png", dpi=fig.dpi)
