import numpy as np


def run(points, points_classification):
	points_matrix = get_points_adapted_matrix(points)

	points_pseudo_inverse = np.dot(np.linalg.inv(np.dot(points_matrix.T, points_matrix)), 
		points_matrix.T)

	return np.dot(points_pseudo_inverse, points_classification).T, points_matrix


def classify_points_given_weights(points_matrix, weights):
	return np.sign(np.dot(points_matrix, weights))


def get_points_adapted_matrix(points):
	return np.matrix([np.insert(point, 0, 1, axis=0) 
		for point in points])
