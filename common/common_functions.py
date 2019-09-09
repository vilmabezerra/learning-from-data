import math

from numpy.linalg import lstsq
from numpy import ones,vstack, random, empty, sign
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


def get_predefined_target(input_number: int):
	return sign(input_number[0]**2 + input_number[1]**2 - 0.6)

def error_surface_result(u, v):
	return (u*math.exp(v) - 2*v*math.exp(-u))**2

def u_partial_derivative(u, v):
	return 2*(math.exp(v) + 2*v*math.exp(-u))*(u*math.exp(v) - 2*v*math.exp(-u))

def v_partial_derivative(u, v):
	return 2*(u*math.exp(v) - 2*math.exp(-u))*(u*math.exp(v) - 2*v*math.exp(-u))

def get_uniform_random_points_classification_with_predefined_target_function(
	uniform_random_points):

	n = uniform_random_points.shape[0]
	points_classification = empty(n)
	for i in range(n):
		point = uniform_random_points[i]
		points_classification[i] = get_predefined_target(point)

	return points_classification

def generate_uniform_random_points_and_their_classes(n_elements=100, y_max=1, y_min=-1):
	points = random.uniform(y_max, y_min, size=(n_elements, 2))

	target_function = get_target_function()
	up_line_classification = random.choice([y_max, y_min])
	points_class = get_uniform_random_points_classification(points, 
		target_function, up_line_classification)

	return points, points_class
