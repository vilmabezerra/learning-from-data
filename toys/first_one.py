from typing import Tuple
from numpy import random, empty, mean
import matplotlib.pyplot as plt

from common.common_functions import get_line_equation
from algorithms import perceptron_learning_algorithm


def get_random_point(random_min=-1, random_max=1):
	x_coord = random.uniform(random_min, random_max)
	y_coord = random.uniform(random_min, random_max)
	return x_coord, y_coord


def get_target_function(random_max=1, random_min=-1) -> Tuple:
	points = list()

	for i in range(2):
		x_coord, y_coord = get_random_point(random_min, random_max)
		points.append((x_coord, y_coord))

	return get_line_equation(points)


def get_target(target_function: Tuple, input_number: int):
	return target_function[0]*input_number + target_function[1]


def get_uniform_random_points_classification(uniform_random_points, 
											 target_function: Tuple, 
											 up_line_classification: int):

	n = uniform_random_points.shape[0]
	points_classification = empty(n)
	for i in range(n):
		point = uniform_random_points[i]
		is_higher_than_target = \
			get_target(target_function, point[0]) < point[1]

		points_classification[i] = up_line_classification \
			if is_higher_than_target \
			else up_line_classification * -1

	return points_classification

def plot_fig(y_min, y_max, uniform_random_points, 
	target_function, uniform_points_classification, 
	pla_line_equation=None):
	fig, ax = plt.subplots(figsize=(6,6))

	axes = plt.gca()
	axes.set_xlim([y_min,y_max])
	axes.set_ylim([y_min,y_max])

	mask = uniform_points_classification == y_max

	ax.scatter(uniform_random_points[:,0][mask], 
		uniform_random_points[:,1][mask], c='purple')
	ax.scatter(uniform_random_points[:,0][~mask], 
		uniform_random_points[:,1][~mask], c='pink')
	ax.plot([y_min, y_max], [
		get_target(target_function, y_min), 
		get_target(target_function, y_max)], 
		c='black')

	if pla_line_equation:
		ax.plot([y_min, y_max], [
		get_target(pla_line_equation, y_min), 
		get_target(pla_line_equation, y_max)], 
		c='red')

	plt.show()

def main_job(**kwargs):
	n_elements = kwargs['n_elements']
	y_min = kwargs['min']
	y_max = kwargs['max']
	times_to_run = kwargs['times_to_run']

	each_time_iterations = list()
	each_time_weights = list()

	for i in range(times_to_run):
		target_function = get_target_function()
		up_line_classification = random.choice([y_max, y_min])

		uniform_random_points = random.uniform(y_min, y_max, size=(n_elements, 2))
		uniform_points_classification = get_uniform_random_points_classification(
			uniform_random_points, target_function, up_line_classification)

		weights, iterations = perceptron_learning_algorithm.run(uniform_random_points, 
			uniform_points_classification)

		pla_random_points = perceptron_learning_algorithm.get_random_points_given_weights(weights)

		pla_line_equation = get_line_equation(pla_random_points)

		# plot_fig(y_min, y_max, uniform_random_points, target_function, 
	 	# 	uniform_points_classification, pla_line_equation)

		each_time_weights.append(weights)
		each_time_iterations.append(iterations)

	print('It took me an average of {} iterations!'.format(
		mean(each_time_iterations)))

	

if __name__ == '__main__':
	main_job(n_elements=10, min=-1, max=1, times_to_run=1000)


