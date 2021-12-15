from vec2 import *
import numpy as np
import sympy as sp

# Has to be here because of the .subs() function which is used in the calculate_* functions
x_i, y_i, x_j, y_j, alpha_i, alpha_j = sp.symbols("x_i y_i x_j y_j alpha_i alpha_j", real=True)
"""
x_j     = x-coordinate of a dipole
y_j     = y-coordinate of a dipole
alpha_j = angle of that dipole w.r.t. the positive x-axis
x_i     = x-coordinate of the dipole w.r.t. we are doing our calculations
y_i     = y-coordinate of the dipole w.r.t. we are doing our calculations
alpha_i = angle (w.r.t. the positive x-axis) of the dipole w.r.t. we are doing our calculations
"""

class Particle():
	# n = number of particles
	n = 0
	# var = NumPy array which stores the list of symbolic variables used by the particle objects
	var = None
	def __init__(self, initial_position, initial_angle):
		# i = particle index
		self.i = Particle.n
		Particle.n += 1

		# initial_position, initial_angle = intial position and angle
		self.initial_position = initial_position
		self.initial_angle = initial_angle

		# symbolic_velocity, symbolic_alpha = symbolic variables unique to the object
		self.symbolic_position = Vec2(sp.symbols("x_" + str(self.i), real=True),
									  sp.symbols("y_" + str(self.i), real=True))
		self.symbolic_alpha = sp.symbols("alpha_" + str(self.i), real=True)

		# Stores the total velocity and angle acting on the object as symbolic expressions
		self.total_velocity = Vec2(0, 0)
		self.total_alpha = 0

		# lambda_position, lambda_velocity, lambda_alpha = lambdified functions.
		# Note: Each coordinate of the lambda Vec2 variables is actually a callable function
		self.lambda_position = Vec2(None, None)
		self.lambda_velocity = Vec2(None, None)
		self.lambda_alpha = None

	# Compute particle position using the relevant equations
	def calculate_plane_velocity(self, particles, expr):
		for j in range(len(particles)):
			if self.i != j:
				other_position = particles[j].symbolic_position
				my_position = self.symbolic_position
				# Defined the next two just to keep up the naming scheme
				other_alpha = particles[j].symbolic_alpha
				my_alpha = self.symbolic_alpha

				self.total_velocity.x += expr.x.subs({x_j: other_position.x, y_j: other_position.y, alpha_j: other_alpha,
													  x_i: my_position.x, y_i: my_position.y, alpha_i: my_alpha})
				self.total_velocity.y += expr.y.subs({x_j: other_position.x, y_j: other_position.y, alpha_j: other_alpha,
													  x_i: my_position.x, y_i: my_position.y, alpha_i: my_alpha})

	# Compute particle angle using the relevant equations
	def calculate_plane_angular_velocity(self, particles, expr):
		for j in range(len(particles)):
			if self.i != j:
				other_position = particles[j].symbolic_position
				my_position = self.symbolic_position
				# Defined the next two just to keep up the naming scheme
				other_alpha = particles[j].symbolic_alpha
				my_alpha = self.symbolic_alpha

				self.total_alpha += expr.subs({x_j: other_position.x, y_j: other_position.y, alpha_j: other_alpha,
											   x_i: my_position.x, y_i: my_position.y, alpha_i: my_alpha})

	@staticmethod
	def get_variable_list(particles):
		"""Computes the list of symbolic variables just once for all objects of this class to speed up the computation.
		   Assumes that the particle list used for all the particles is the same. (Reasonable assumption)"""
		if Particle.var is None:
			n = len(particles)
			Particle.var = np.empty(3 * n, dtype=np.dtype(sp.core.symbol.Symbol))
			for j in range(n):
				Particle.var[2 * j] = particles[j].symbolic_position.x
				Particle.var[2 * j + 1] = particles[j].symbolic_position.y
				Particle.var[2 * n + j] = particles[j].symbolic_alpha

	# Lambdified symbolic functions are faster for numerical calculations.
	# I used this approach (compute first symbolic equations of motion and then compile the function with lambdify)
	# to avoid python loops in the vectorfield function which needs to be run thousands of times and that is slow.
	def lambdify_position(self):
		self.lambda_position.x = sp.lambdify(self.symbolic_position.x, self.symbolic_position.x, modules="numpy")
		self.lambda_position.y = sp.lambdify(self.symbolic_position.y, self.symbolic_position.y, modules="numpy")

	def lambdify_velocity(self, particles):
		"""Lambdifies the expression for the particle velocity, using the list of symbolic variables
		   obtained from `get_variable_list`."""
		# var = []
		# for j in range(n):
			# var.append(particles[j].symbolic_position.x)
			# var.append(particles[j].symbolic_position.y)

		# for j in range(n):
		# 	var.append(particles[j].symbolic_alpha)
		Particle.get_variable_list(particles)
		# print(Particle.var)

		self.lambda_velocity.x = sp.lambdify([Particle.var], self.total_velocity.x, modules="numpy")
		self.lambda_velocity.y = sp.lambdify([Particle.var], self.total_velocity.y, modules="numpy")

	def lambdify_alpha(self, particles):
		"""Lambdifies the expression for alpha, using the list of symbolic variables
		   obtained from `get_variable_list`."""
		Particle.get_variable_list(particles)
		self.lambda_alpha = sp.lambdify([Particle.var], self.total_alpha, modules="numpy")
