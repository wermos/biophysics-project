from math import sqrt

class Vec2:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	# Used for debugging. This method is called when you print an instance
	def __str__(self):
		return f"({self.x}, {self.y})"

	def __add__(self, v):
		return Vec2(self.x + v.x, self.y + v.y)

	def __radd__(self, v):
		return Vec2(self.x + v.x, self.y + v.y)

	def __sub__(self, v):
		return Vec2(self.x - v.x, self.y - v.y)

	def __rsub__(self, v):
		return Vec2(v.x - self.x , v.y - self.y)

	def __mul__(self, n):
		return Vec2(self.x * n, self.y * n)

	def __rmul__(self, n):
		return Vec2(self.x * n, self.y * n)

	def dot(self, v):
		return self.x * v.x + self.y * v.y

	def get_length(self):
		# Note that math.sqrt is faster than np.sqrt on scalars.
		return sqrt(self.dot(self))

	# def subs(self, dictionary):
	# 	return Vec2(self.x.subs(dictionary), self.y.subs(dictionary))
