#!/usr/bin/python

from numpy import linspace, sin, cos, pi, gradient, linalg, power, add, rad2deg, deg2rad, arange
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DoublePendulum:
	"""Class for double pendulum system.
	- Computes motion of system over time
	- Computes properties of system:
		position: t, x, y, theta1, theta2
		velocity: vx, vy, omega1, omega2
		energy: Kinetic, potential and total energy 
	- Creates animation of motion of double pendulum	
	
	"""
	def __init__(self, l1=1, m1=1, l2=1, m2=1, g=9.81):
		"""
		Inputs:
			l1: Length of pendulum 1
			l2: Length of pendulum 2
			m1: Mass of pendulum 1
			m2: Mass of pendulum 2
			g: acceleration of gravity
			 
		"""
		self.l1 = l1
		self.m1 = m1
		self.l2 = l2
		self.m2 = m2
		self.g = g
		self.sol = []
		self.solved = False
		self.dt = 0
		self.angles = "rad"

	def __call__(self, t, y):
		"""
		Inputs:
			t: time
			y: initial values of position and velocity. Must be a list or tuple of radians.
			
		Returns the differantiated positions and velocities of the pendulums (theta1, omega1, theta2, omega2)
		
		"""
		g = self.g
		l1 = self.l1
		l2 = self.l2
		m1 = self.m1
		m2 = self.m2
		theta1 = y[0]
		omega1 = y[1]
		theta2 = y[2]
		omega2 = y[3]
		
		delta = theta2 - theta1
		
		dtheta1 = omega1
		
		den1 = (m1 + m2)*l1 - m2*l1*cos(delta)*cos(delta)
		domega1 = (m2*l1*omega1*omega1*sin(delta)*cos(delta) +
							m2*g*sin(theta2)*cos(delta) +
							m2*l2*omega2*omega2*sin(delta) -
							(m1 + m2)*g*sin(theta1))/den1
		
		dtheta2 = omega2
		
		den2 = (l2/l1)*den1
		
		domega2 = (-m2*l2*omega2*omega2*sin(delta)*cos(delta) + 
							(m1 + m2)*g*sin(theta1)*cos(delta) - 
							(m1 + m2)*l1*omega1*omega1*sin(delta) - 
							(m1 + m2)*g*sin(theta2))/den2
	
		return (dtheta1, domega1, dtheta2, domega2)


	def solve(self, u0, T, dt, angles="rad"):
		""" Solves for the motion of the system by calling __call__.
		
		Input:
			u0: Initial values of the state of motion
			T: Endtime
			dt: time step
			angles: specifies if input is in radians of degrees. Radians by default.
			
		"""
		self.solved = True
		t_values = linspace(0, T, T/dt+1)
		if angles == "deg":
			u0 = deg2rad(u0)
			self.angles = "deg"
		self.sol = solve_ivp(self, [0, T], u0, t_eval=t_values, method="Radau")
		self.dt = dt
		self.time = T
		#return self.sol.t, self.sol.y		

	@property
	def t(self):
		"""Returns array of times."""
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		return self.sol.t
	
	@property
	def theta1(self):
		"""Returns array of angles of position of pendulum 1."""
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		if self.angles == "deg":
			return rad2deg(self.sol.y[0])
		else:
			return self.sol.y[0]
	
	@property
	def omega1(self):
		"""Returns array of angles of velocity of pendulum 2."""
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		if self.angles == "deg":
			return rad2deg(self.sol.y[1])
		return self.sol.y[1]
		
	@property
	def theta2(self):
		"""Returns array of angles of position of pendulum 2."""
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		if self.angles == "deg":
			return rad2deg(self.sol.y[2])
		return self.sol.y[2]
		
	@property
	def omega2(self):
		"""Returns array of angles of velocity of pendulum 2."""
		if not self.solved:
			raise RuntimeError("The system is not solved.")
		if self.angles == "deg":
			return rad2deg(self.sol.y[3])
		return self.sol.y[3]
	
	@property
	def x1(self):
		"""Returns array of x-values of pendulum 1 """
		l1 = self.l1
		return [l1*sin(i) for i in self.sol.y[0]]
		
	@property
	def y1(self):
		"""Returns array of y-values of pendulum 1 """
		l1 = self.l1
		return [-l1*cos(i) for i in self.sol.y[0]]
		
	@property
	def x2(self):
		"""Returns array of x-values of pendulum 2 """
		l2 = self.l2
		return add(self.x1, [l2*sin(i) for i in self.sol.y[2]])
		
	@property
	def y2(self):
		"""Returns array of y-values of pendulum 2 """
		l2 = self.l2
		return add(self.y1, [- l2*cos(i) for i in self.sol.y[2]])
		
	@property
	def potential1(self):
		"""Returns the potential energy of pendulum 1"""
		m1 = self.m1
		l1 = self.l1
		g = self.g
		y1 = self.y1
		return [m1*g*(i + l1) for i in y1]
	
	@property
	def potential2(self):
		"""Returns the potential energy of pendulum 2"""
		m2 = self.m2
		l1 = self.l1
		l2 = self.l2
		g = self.g
		y2 = self.y2
		return [m2*g*(i + l1 + l2) for i in y2]
	
	@property
	def potential(self):
		"""Returns the potential energy of the system"""
		return add(self.potential1, self.potential2)

	@property
	def vx1(self):
		return gradient(self.x1, self.dt)
		
	@property
	def vy1(self):
		return gradient(self.y1, self.dt)
		
	@property
	def vx2(self):
		return gradient(self.x2, self.dt)
		
	@property
	def vy2(self):
		return gradient(self.y2, self.dt)
	
	@property
	def kinetic1(self):
		"""Returns the kinetic energy of pendulum 1"""
		m1 = self.m1
		k = lambda u, v, m: 0.5 * m * add(power(u, 2), power(v, 2))
		return k(self.vx1, self.vy1, m1) 

	@property
	def kinetic2(self):
		"""Returns the kinetic energy of pendulum 2"""
		m2 = self.m2
		k = lambda u, v, m: 0.5 * m * add(power(u, 2), power(v, 2))
		return k(self.vx2, self.vy2, m2) 
	
	@property
	def kinetic(self):
		"""Returns the kinetic energy of the system"""
		return add(self.kinetic1, self.kinetic2)
	
	@property
	def total(self):
		"""Returns the total energy of the system"""
		return add(self.kinetic, self.potential)
	
	def create_animation(self):
		"""Creates animation of the motion of the double pendulum system. """
		fig = plt.figure()
		dim = self.l1 + self.l2 + 2
		ax = fig.add_subplot(111, autoscale_on=False, xlim=(-dim, dim), ylim=(-dim, dim))
		
		plt.title("Animation of Double Pendulum with Traces")
		ax.set_aspect('equal')
		plt.axis("off")
		ax.grid()
		
		self.line, = ax.plot([], [], 'o-', lw=2, color="blue")
		self.tr1, = ax.plot([], [], color="blue", marker="o", alpha = 0.2)
		self.tr2, = ax.plot([], [], color="blue", marker="o", alpha = 0.2)
		self.tr3, = ax.plot([], [], color="blue", marker="o", alpha = 0.4)
		self.tr4, = ax.plot([], [], color="blue", marker="o", alpha = 0.4)
		
		self.time_template = 'time = %.1fs'
		self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
	
		self.delay = 6
		self.offset = 3
		self.traces1 = [(self.x1[0], self.y1[0]) for i in range(self.delay)]
		self.traces2 = [(self.x2[0], self.y2[0]) for i in range(self.delay)]
		self.traces3 = [(self.x1[0], self.y1[0]) for i in range(self.offset)]
		self.traces4 = [(self.x2[0], self.y2[0]) for i in range(self.offset)]
		
		
		for i in range(len(self.x1)-self.delay):
			self.traces1.append((self.x1[i], self.y1[i]))
			self.traces2.append((self.x2[i], self.y2[i]))
	
		for i in range(len(self.x1)-(self.delay-self.offset)):
			self.traces3.append((self.x1[i], self.y1[i]))
			self.traces4.append((self.x2[i], self.y2[i]))
	
		self.ani = animation.FuncAnimation(fig, self.animate, arange(1, len(self.y1)),
									  interval=(1000*self.dt), blit=True, init_func=self.init)
		
			
	def init(self):
		self.line.set_data([], [])
		self.time_text.set_text('')
		return self.line, self.time_text
	
	def animate(self, i):
		thisx = [0, self.x1[i], self.x2[i]]
		thisy = [0, self.y1[i], self.y2[i]]
		
		self.tr1.set_data(self.traces1[i])
		self.tr2.set_data(self.traces2[i])
		self.tr3.set_data(self.traces3[i])
		self.tr4.set_data(self.traces4[i])
	
		self.line.set_data(thisx, thisy)
		self.time_text.set_text(self.time_template % (i*self.dt))
		return self.line, self.time_text, self.tr1, self.tr2, self.tr3, self.tr4	
	

	def show_animation(self):
		"""Shows animation on screen."""
		plt.show()
	
	def save_animation(self, filename):
		"""Saves animation to file.
		
		Input: name of file
		
		"""
		self.ani.save(filename, fps=60)
		

if __name__ == "__main__":
	#initialising DoublePendulum system						
	d_pen = DoublePendulum(4, 3, 2.5, 2)
	y1 = (pi/2, 0, pi/6, -pi/2)
	dt = 1/60

	#Solving motion of system
	d_pen.solve(y1, 10, dt)
	
	#Plotting kinetic, potential and total energy
	plt.plot(d_pen.t, d_pen.potential, label="Potential")
	plt.plot(d_pen.t, d_pen.kinetic, label="Kinetic")
	plt.plot(d_pen.t, d_pen.total, label="Total")
	plt.legend()
	plt.title("Potential, Kinetic and Total Energy")
	plt.show()
	
	#Create animation and storing it to file
	d_pen.create_animation()
	d_pen.show_animation()
	d_pen.save_animation("example_simulation.mp4")


