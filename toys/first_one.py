from typing import Tuple
from numpy import random, empty, mean
import matplotlib.pyplot as plt

from common.common_functions import get_line_equation, get_target_function,\
	get_uniform_random_points_classification
from algorithms import perceptron_learning_algorithm


def calculate_expected_error(n_elements, pla_function, target_function, 
	up_line_classification, y_min, y_max):

	other_random_points = random.uniform(y_min, y_max, size=(n_elements, 2))
	other_points_classification_pla = get_uniform_random_points_classification(
		other_random_points, pla_function, up_line_classification)
	other_points_classification_target = get_uniform_random_points_classification(
		other_random_points, target_function, up_line_classification)

	misclassified_points_quantity = sum([1 for i, x in enumerate(
 		other_points_classification_pla) if x != other_points_classification_target[i]])

	return misclassified_points_quantity/n_elements

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
	each_time_expected_error = list()

	for i in range(times_to_run):
		target_function = get_target_function()
		up_line_classification = random.choice([y_max, y_min])

		uniform_random_points = random.uniform(y_min, y_max, size=(n_elements, 2))
		uniform_points_classification = get_uniform_random_points_classification(
			uniform_random_points, target_function, up_line_classification)

		weights, iterations = perceptron_learning_algorithm.run(
			uniform_random_points, uniform_points_classification)

		pla_random_points = \
			perceptron_learning_algorithm.get_random_points_given_weights(weights)

		pla_line_equation = get_line_equation(pla_random_points)

		# plot_fig(y_min, y_max, uniform_random_points, target_function, 
	 	# 	uniform_points_classification, pla_line_equation)
		expected_error = calculate_expected_error(n_elements, pla_line_equation, 
			target_function, up_line_classification, y_min, y_max)

		each_time_expected_error.append(expected_error)
		each_time_iterations.append(iterations)

	print('It took me an average of {} iterations!'.format(
		mean(each_time_iterations)))

	print('Expected error: {}'.format(
		mean(each_time_expected_error)))

	

if __name__ == '__main__':
	main_job(n_elements=10, min=-1, max=1, 
		times_to_run=1000)


