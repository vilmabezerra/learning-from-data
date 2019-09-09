import numpy as np
import math
from common.common_functions import *
from algorithms.perceptron_learning_algorithm import *

def error_function(points, weights, points_class):
	error = np.log(
		1 + np.exp(np.dot(points, weights)*(-points_class))
		) 
	error = np.sum(error) / len(points)
	return error

def error_function_gradient(point, weights, point_class):
	gradient = -(point_class*point) / (1+ np.exp(point_class*np.inner(weights, point)))
	return gradient

def calculate_logistic_regression_e_out(weights, n_elements, y_min, y_max):
	test_points, test_points_class = generate_uniform_random_points_and_their_classes(
		n_elements, y_min, y_max)
	test_points = np.insert(test_points, 0, 1, axis=1) 
	e_out = error_function(test_points, weights, test_points_class,)
	return e_out

def logistic_regression(n_elements=100, y_min=-1, y_max=1, learning_rate=0.01):
	points, points_class = generate_uniform_random_points_and_their_classes(
		n_elements, y_min, y_max)
	points = np.insert(points, 0, 1, axis=1) 
	weights = np.zeros(3)

	epochs = 0
	weights_diff = math.inf
	previous_weights = np.zeros(3)
	while weights_diff > 0.01:
		for i in np.random.permutation(len(points)):
			gradient = error_function_gradient(points[i], weights, points_class[i])
			weights -= gradient*learning_rate

		epochs += 1
		weights_diff = np.linalg.norm(previous_weights - weights)
		previous_weights = weights.copy()
		
		# pads = 100
		# output_str = f'\repoch: {epochs}, weights_diff: {weights_diff}'
		# print(output_str, end=' '*(pads-len(output_str)))

	return epochs, weights
