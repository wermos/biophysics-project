from vec2 import *
import numpy as np
import sympy as sp

# Has to be here because of the .subs() function
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
	var = None
	def __init__(self, initial_position, initial_angle):
		# i = particle index
		self.i = Particle.n
		Particle.n += 1

		# initial_position, initial_angle = intial position and angle
		self.initial_position = initial_position
		self.initial_angle = initial_angle

		# position, velocity, alpha = symbolic variables
		self.symbolic_position = Vec2(sp.symbols("x_" + str(self.i), real=True), sp.symbols("y_" + str(self.i), real=True))
		self.symbolic_alpha = sp.symbols("alpha_" + str(self.i), real=True)

		# Stores the total force and torque on the object as symbolic expressions
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
				self.total_velocity.x += expr.x.subs({x_j: particles[j].symbolic_position.x, y_j: particles[j].symbolic_position.y, alpha_j: particles[j].symbolic_alpha,
													  x_i: self.symbolic_position.x, y_i: self.symbolic_position.y, alpha_i: self.symbolic_alpha})
				self.total_velocity.y += expr.y.subs({x_j: particles[j].symbolic_position.x, y_j: particles[j].symbolic_position.y, alpha_j: particles[j].symbolic_alpha,
													  x_i: self.symbolic_position.x, y_i: self.symbolic_position.y, alpha_i: self.symbolic_alpha})

	# Compute particle angle using the relevant equations
	def calculate_plane_angular_velocity(self, particles, expr):
		for j in range(len(particles)):
			if self.i != j:
				self.total_alpha += expr.subs({x_j: particles[j].symbolic_position.x, y_j: particles[j].symbolic_position.y, alpha_j: particles[j].symbolic_alpha,
											   x_i: self.symbolic_position.x, y_i: self.symbolic_position.y, alpha_i: self.symbolic_alpha})

	# lambdified symbolic functions are faster for numerical calculations.
	# I used this approach (compute first symbolic equations of motion and then compile the function with lambdify)
	# to avoid python loops in the vectorfield function which needs to be run thousands of times and that is slow.
	@staticmethod
	def get_variable_list(particles):
		"""Assumes that the particle list used for all the particles is the same. (Reasonable assumption)"""
		if Particle.var is None:
			n = len(particles)
			Particle.var = np.empty(3 * n, dtype=np.dtype(sp.core.symbol.Symbol))
			for j in range(n):
				Particle.var[2 * j] = particles[j].symbolic_position.x
				Particle.var[2 * j + 1] = particles[j].symbolic_position.y
				Particle.var[2 * n + j] = particles[j].symbolic_alpha

	def lambdify_position(self):
		self.lambda_position.x = sp.lambdify(self.symbolic_position.x, self.symbolic_position.x)
		self.lambda_position.y = sp.lambdify(self.symbolic_position.y, self.symbolic_position.y)

	def lambdify_velocity(self, particles):
		# var = []
		# for j in range(n):
			# var.append(particles[j].symbolic_position.x)
			# var.append(particles[j].symbolic_position.y)

		# for j in range(n):
		# 	var.append(particles[j].symbolic_alpha)
		Particle.get_variable_list(particles)
		# print(Particle.var)

		self.lambda_velocity.x = sp.lambdify([Particle.var], self.total_velocity.x)
		self.lambda_velocity.y = sp.lambdify([Particle.var], self.total_velocity.y)

	def lambdify_alpha(self, particles):
		Particle.get_variable_list(particles)
		self.lambda_alpha = sp.lambdify([Particle.var], self.total_alpha)
