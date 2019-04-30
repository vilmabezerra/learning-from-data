from typing import List, Tuple
from numpy import ones,vstack, random, empty
from numpy.linalg import lstsq
import matplotlib.pyplot as plt


def get_line_equation(points: List) -> Tuple:
	x_coords, y_coords = zip(*points)
	A = vstack([x_coords,ones(len(x_coords))]).T
	m, c = lstsq(A, y_coords, rcond=None)[0]
	return (m, c)


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
		is_higher_than_target = get_target(target_function, point[0]) < point[1]
		points_classification[i] = up_line_classification \
			if is_higher_than_target \
			else up_line_classification * -1

	return points_classification

def plot_fig(y_min, y_max, uniform_random_points, 
	target_function, uniform_points_classification):
	fig, ax = plt.subplots(figsize=(6,6))

	axes = plt.gca()
	axes.set_xlim([y_min,y_max])
	axes.set_ylim([y_min,y_max])

	mask = uniform_points_classification == y_max

	ax.scatter(uniform_random_points[:,0][mask], uniform_random_points[:,1][mask], c='purple')
	ax.scatter(uniform_random_points[:,0][~mask], uniform_random_points[:,1][~mask], c='pink')
	ax.plot([y_min, y_max], [
		get_target(target_function, y_min), 
		get_target(target_function, y_max)], 
		c='yellow')

	plt.show()

def main_job(**kwargs):
	n_elements = kwargs['n_elements']
	y_min = kwargs['min']
	y_max = kwargs['max']

	target_function = get_target_function()
	up_line_classification = random.choice([y_max, y_min])

	uniform_random_points = random.uniform(y_min, y_max, size=(n_elements, 2))
	uniform_points_classification = get_uniform_random_points_classification(
		uniform_random_points, target_function, up_line_classification)

	plot_fig(y_min, y_max, uniform_random_points, target_function, 
		uniform_points_classification)


if __name__ == '__main__':
	main_job(n_elements=100, min=-1, max=1)


