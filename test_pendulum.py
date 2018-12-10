#!/usr/bin/python

import nose.tools as nt
from pendulum import Pendulum
from math import pi
from numpy import linspace, add, power

def test_stationary():
	var = Pendulum(l=2.2)
	theta0 = 0
	omega0 = 0
	y0 = [theta0, omega0]
	y = var(0, y0)
	nt.assert_equal(y, (0.0, 0.0))

def test_call():
	var = Pendulum(l=2.2)
	theta0 = pi/4
	omega0 = 0.1
	y0 = [theta0, omega0]
	y = var(0, y0)
	akk = 3
	sol = (round(y[0], akk), round(y[1], akk))
	nt.assert_equal(sol, (0.1, -3.1530))
		
@nt.raises(RuntimeError)
def test_exceptions():
	var = Pendulum(l=2.2)
	var.t
	var.theta
	var.omega
	
def test_zeroes():
	var = Pendulum(l=2.2)
	T = 5
	dt = 0.1
	var.solve((0, 0), T, dt)
	time = linspace(0, T, T/dt+1)
	nt.assert_equal(time.all(), var.t.all())
	nt.assert_equal(0, var.theta.all())
	nt.assert_equal(0, var.omega.all())
	
def test_length():
	L = 3
	var = Pendulum(l=L)
	T = 5
	dt = 0.1
	var.solve((pi/4, 0.1), T, dt)
	success = True
	for i in range(len(var.x)):
		if var.x[i]**2 + var.y[i]**2 - L**2 > 0.000000001:
			success = False
	nt.assert_true(success)
	
	