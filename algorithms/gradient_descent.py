import numpy as numpy
from common.common_functions import *


def update_variable_given_step_size(variable, step_size):
	variable -= step_size
	return variable

def get_variable_step_size(learning_rate, derivative):
	return learning_rate*derivative

def gradient_descent_step(u, v, learning_rate):
	u_derivative = u_partial_derivative(u, v)
	v_derivative = v_partial_derivative(u, v)
	u_step_size = get_variable_step_size(learning_rate, u_derivative)
	v_step_size = get_variable_step_size(learning_rate, v_derivative)
	u = update_variable_given_step_size(u, u_step_size)
	v = update_variable_given_step_size(v, v_step_size)
	actual_value = error_surface_result(u, v)
	return actual_value, u, v

def coordinate_descent_step(u, v, learning_rate, iteration_number):
	if iteration_number%2:
		v_derivative = v_partial_derivative(u, v)
		step_size = get_variable_step_size(learning_rate, v_derivative)
		v = update_variable_given_step_size(v, step_size)
	else:
		u_derivative = u_partial_derivative(u, v)
		step_size = get_variable_step_size(learning_rate, u_derivative)
		u = update_variable_given_step_size(u, step_size)
	actual_value = error_surface_result(u, v)
	return actual_value, u, v
