from algorithms.gradient_descent import *
from algorithms.logistic_regression import *


def iterations_to_achieve_minimum_error(min_value, initial_u=1, initial_v=1, learning_rate=0.1):
	n_iterations = 1
	u = initial_u
	v = initial_v
	actual_value = error_surface_result(u, v)
	while actual_value > min_value:
		actual_value, u, v = gradient_descent_step(u, v, learning_rate)
		n_iterations += 1
	return n_iterations, u, v

def minimum_error_given_iterations_number(iterations, initial_u=1, initial_v=1, learning_rate=0.1):
	u = initial_u
	v = initial_v
	actual_value = error_surface_result(u, v)
	while iterations > 0:
		actual_value, u, v = coordinate_descent_step(u, v, learning_rate, iterations)
		iterations -= 1
	return actual_value, u, v

def average_epochs_and_e_out_to_converge(iterations, n_elements, y_min, y_max, learning_rate):
	sum_epochs = []
	e_outs = []
	n_iterations = iterations
	while n_iterations > 0:
		epochs, weights = logistic_regression(n_elements, y_min, y_max, learning_rate)
		sum_epochs.append(epochs)

		e_out = calculate_logistic_regression_e_out(weights, n_elements, y_min, y_max)
		e_outs.append(e_out)
		n_iterations -= 1 

	average_epochs = sum(sum_epochs) / iterations
	average_e_out = sum(e_outs) / iterations
	return average_epochs, average_e_out

def gradient_descent_part(min_value):
	n_iterations, u, v = \
		iterations_to_achieve_minimum_error(min_value)

	print('\nIterations to error get bellow to 1.0e-14: {}'.format(n_iterations))
	print(f'\nu and v coordinates at the end: ({u}, {v})')

	min_error, u, v = minimum_error_given_iterations_number(iterations=30)
	print('\nError after 15 full iterations: {:.3e}'.format(min_error))

def logistic_regression_part(iterations, n_elements=100, y_min=-1, y_max=1, learning_rate=0.01):
	average_epochs, average_e_out = \
		average_epochs_and_e_out_to_converge(iterations, n_elements, y_min, y_max, learning_rate)

	print('\nAverage epochs to converge: {}'.format(average_epochs))
	print('\nAverage e_out: {}'.format(average_e_out))


if __name__ == '__main__':
	gradient_descent_part(min_value=1e-14)
	logistic_regression_part(iterations=100)
