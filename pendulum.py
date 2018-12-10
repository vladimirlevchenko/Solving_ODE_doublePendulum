#!/usr/bin/python
from numpy import linspace, sin, cos, pi, gradient, linalg, power, add, deg2rad, rad2deg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Pendulum:
	'''
	Class for pendulum object and determining its movement over time.
	- computes potential, kinetic and total energy with plots.
	- properties: theta, omega, t, x, y, dx/dt, dy/dt, kinetic, potential and total energy.
	'''
	def __init__(self, l=1, m=1, g=9.81):
		"""
		Inputs:
			l: Length of pendulum
			m: Mass of pendulum
			g: gravity
			
		"""
		self.l = l
		self.m = m
		self.g = g
		self.sol = []
		self.solved = False
		self.dt = 0
		self.angles = "rad"
	
	def __call__(self, t, y):
		"""
		Returns dtheta / dt and domega / dt
		
		"""
		g = self.g
		l = self.l
		theta = y[0]
		omega = y[1]
		
		dtheta = omega
		domega = -g/l*sin(theta)
		
		return (dtheta, domega)
	
	def solve(self, u0, T, dt, angles="rad"):
		"""
		Uses solve_ivp to return solutions to call. Stores solutions in attribute sol.
		
		"""
		self.solved = True
		t_values = linspace(0, T, T/dt+1)
		if angles == "deg":
			u0 = deg2rad(u0)
			self.angles = "deg"
		self.sol = solve_ivp(self, [0, T], u0, t_eval=t_values)
		self.dt = dt
	
	@property
	def t(self):
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		return self.sol.t
	
	@property
	def theta(self):
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		elif self.angles == "deg":
			return rad2deg(self.sol.y[0])
		return self.sol.y[0]
	
	@property
	def omega(self):
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		elif self.angles == "deg":
			return rad2deg(self.sol.y[1])
		return self.sol.y[1]
	
	@property
	def x(self):
		l = self.l
		return [l*sin(i) for i in self.theta]
		
	@property
	def y(self):
		l = self.l
		return [-l*cos(i) for i in self.theta]
		
	@property
	def potential(self):
		m = self.m
		l = self.l
		g = self.g
		y = self.y
		return [m*g*(i + l) for i in y]
		
	@property
	def vx(self):
		return gradient(self.x, self.dt)
	
	@property
	def vy(self):
		return gradient(self.y, self.dt)
		
	@property
	def kinetic(self):
		m = self.m
		vx = self.vx
		vy = self.vy
		k = lambda x, y: 0.5 * m * add(power(x, 2), power(y, 2))
		return k(self.vx, self.vy) 
	
	@property
	def total(self):
		return add(self.kinetic, self.potential)
		

class DampenedPendulum(Pendulum):
	def __init__(self, l=1, m=1, g=9.81, b=0):
		self.l = l
		self.m = m
		self.g = g
		self.b = b
		self.sol = []
		self.solved = False
		self.dt = 0
		self.angles = "rad"

	def __call__(self, t, y):
		g = self.g
		l = self.l
		b = self.b
		m = self.m
		theta = y[0]
		omega = y[1]
		
		dtheta = omega
		domega = -g/l*sin(theta) - b/float(m)*omega
				
		return (dtheta, domega)
		
		
if __name__ == "__main__":	
	# Creating instance of Pendulum class
	pen = Pendulum(2.2)
	y0 = (pi/4, 0.2)
	
	#Solve for the motion of the pendulum.
	pen.solve(y0, 10, 0.01)
	
	#Plotting motion over time.
	plt.figure()
	plt.plot(pen.t, pen.theta)
	plt.title("Motion of Pendulum")
	plt.xlabel("Time")
	plt.ylabel("Angle (radians)")
	
	#Plotting energy
	plt.figure()
	plt.title("Energy of Pendulum")
	plt.xlabel("Time")
	plt.ylabel("Energy")
	plt.plot(pen.t, pen.kinetic, label="Kinetic")
	plt.plot(pen.t, pen.potential, label="Potential")
	plt.plot(pen.t, pen.total, label="Total")
	plt.legend()

	# Plotting instance of DampenedPendulum 
	dam_pen = DampenedPendulum(10, b=0.05)
	dam_pen.solve(y0, 10, 0.1)
	plt.figure()
	plt.title("Energy of DampenedPendulum")
	plt.xlabel("Time")
	plt.ylabel("Energy")
	plt.plot(dam_pen.t, dam_pen.total, label="Total Energy")
	plt.legend()
	plt.show()
	
		