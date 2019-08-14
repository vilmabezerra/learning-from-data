import math

from algorithms import gradient_descent

def error_surface_result(u, v):
	return (u*math.exp(v) - 2*v*math.exp(-u))**2

def u_partial_derivative(u, v):
	return 2*(math.exp(v) + 2*v*math.exp(-u))*(u*math.exp(v) - 2*v*math.exp(-u))

def v_partial_derivative(u, v):
	return 2*(u*math.exp(v) - 2*math.exp(-u))*(u*math.exp(v) - 2*v*math.exp(-u))


def gradient_step(u, v, learning_rate):
	u_derivative = u_partial_derivative(u, v)
	v_derivative = v_partial_derivative(u, v)
	u_step_size = gradient_descent.get_variable_step_size(learning_rate, u_derivative)
	v_step_size = gradient_descent.get_variable_step_size(learning_rate, v_derivative)
	u = gradient_descent.update_variable_given_step_size(u, u_step_size)
	v = gradient_descent.update_variable_given_step_size(v, v_step_size)
	actual_value = error_surface_result(u, v)
	return actual_value, u, v

def iterations_to_achieve_minimum_error(min_value, initial_u=1, initial_v=1, learning_rate=0.1):
	n_iterations = 1
	u = initial_u
	v = initial_v
	actual_value = error_surface_result(u, v)
	while actual_value > min_value:
		actual_value, u, v = gradient_step(u, v, learning_rate)
		n_iterations += 1
	return n_iterations, u, v

def minimum_error_given_iterations_number(iterations, initial_u=1, initial_v=1, learning_rate=0.1):
	u = initial_u
	v = initial_v
	actual_value = error_surface_result(u, v)
	while iterations > 0:
		actual_value, u, v = gradient_step(u, v, learning_rate)
		iterations -= 1
	return actual_value, u, v

def gradient_descent_part(min_value):
	n_iterations, u, v = \
		iterations_to_achieve_minimum_error(min_value)

	print('Iterations to error get bellow to 1e-14: {}'.format(n_iterations))
	print(f'u and v coordinates at the end: ({u}, {v})')

	min_error = minimum_error_given_iterations_number(iterations=15)
	print(f'Error after 15 iterations: {min_error}')




if __name__ == '__main__':
	gradient_descent_part(min_value=1e-14)
