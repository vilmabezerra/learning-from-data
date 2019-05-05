from numpy.linalg import lstsq
from numpy import ones,vstack, random, empty
from typing import List, Tuple


def get_line_equation(points: List) -> Tuple:
	x_coords, y_coords = zip(*points)
	A = vstack([x_coords,ones(len(x_coords))]).T
	m, c = lstsq(A, y_coords, rcond=None)[0]
	return (m, c)

def get_random_point(random_min=-1, random_max=1):
	x_coord = random.uniform(random_min, random_max)
	y_coord = random.uniform(random_min, random_max)
	return x_coord, y_coord


def get_target(target_function: Tuple, input_number: int):
	return target_function[0]*input_number + target_function[1]


def get_target_function(random_max=1, random_min=-1) -> Tuple:
	points = list()

	for i in range(2):
		x_coord, y_coord = get_random_point(random_min, random_max)
		points.append((x_coord, y_coord))

	return get_line_equation(points)

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
