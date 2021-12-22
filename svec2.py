from math import pi

# This is the spherical vector class which does coordinate-wise addition, modulo pi and 2 * pi respectively 

class SVec2:
	def __init__(self, theta, phi):
		self.theta = theta
		self.phi = phi

	# Used for debugging. This method is called when you print an instance
	def __str__(self):
		return f"({self.theta}, {self.phi})"

	def __add__(self, v):
		return SVec2((self.theta + v.theta) % pi, (self.phi + v.phi) % (2 * pi))

	def __radd__(self, v):
		return SVec2((self.theta + v.theta) % pi, (self.phi + v.phi) % (2 * pi))

	def __sub__(self, v):
		return SVec2((self.theta - v.theta) % pi, (self.phi - v.phi) % (2 * pi))

	def __rsub__(self, v):
		return SVec2((v.theta - self.theta) % pi, (v.phi - self.phi) % (2 * pi))

	def __mul__(self, n):
		return SVec2(self.theta * n, self.phi * n)

	def __rmul__(self, n):
		return SVec2(self.theta * n, self.phi * n)

	def dot(self, v):
		return self.theta * v.theta + self.phi * v.phi

	
