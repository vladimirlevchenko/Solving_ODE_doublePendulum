#!/usr/bin/python

import nose.tools as nt
from double_pendulum import DoublePendulum
from numpy import linspace

def test_stationary():
	"""Test if stationary initial state result in stationary behavior."""
	var = DoublePendulum(2, 2, 2, 2)
	y0 = [0, 0, 0, 0]
	y = var(0, y0)
	nt.assert_equal(y, (0.0, 0.0, 0.0, 0.0))
	
@nt.raises(RuntimeError)
def test_exceptions():
	"""Test if expected exceptions are raised."""
	var = DoublePendulum(2, 2, 2, 2)
	var.t
	var.theta1
	var.theta2
	var.omega1
	var.omega2
	
def test_zeroes():
	"""Test if solved values are zeroes if initial state is stationary."""
	var = DoublePendulum(2, 2, 2, 2)
	T = 5
	dt = 0.1
	var.solve((0, 0, 0, 0), T, dt)
	time = linspace(0, T, T/dt+1)
	nt.assert_equal(time.all(), var.t.all())
	nt.assert_equal(0, var.theta1.all())
	nt.assert_equal(0, var.theta2.all())
	nt.assert_equal(0, var.omega1.all())
	nt.assert_equal(0, var.omega2.all())