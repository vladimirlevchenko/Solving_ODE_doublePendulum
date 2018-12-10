import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class ExponentialDecay():
    """solves expeonential decay ODE
    Equation:
        du/dt = -a * u
    """
    def __init__(self, a):
        """
		Args:
			a: decay constant

		"""
        self.a = a

    def __call__(self, t, u):
        """Differantiate function for linear decay.

		Args:
			t: time
			u: quantity to be investigated
			return: differentiation of function du / dt

		"""
        return -self.a*u

    def solve(self, u0, T, dt):
        """Solves ODE

		Uses solve_ivp from scitools.integrate to solve ODE.
		Throws exception if input is not list or tuple.

		Args:
			u0: List of tuple of initial conditions.
			T: Stop
			dt: timestep
			return: t, y from solve_ivp. Time and list of solutions.

		"""
        if type(u0) != (list or tuple):
            raise ValueError('Input is not list or tuple')
        t = np.arange(0, T, dt)
        t_eval = np.arange(0, T, dt)
        solution = solve_ivp(self, [0,T], u0, t_eval=t_eval)
        return solution.t, solution.y

#Main block

if __name__ == '__main__':
    decay_model = ExponentialDecay(0.4)
    u0 = [3.4,6,8,23]
    t,u = decay_model.solve(u0, 12, 0.1)

    for i in range(len(u0)):
        plt.plot(t, u[i])
    plt.legend(u0)
    plt.xlabel('time')
    plt.ylabel('decay')
    plt.title('Exponential Decay')
    plt.show()
