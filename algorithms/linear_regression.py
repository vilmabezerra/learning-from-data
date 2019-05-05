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


def get_transformed_points_matrix(points):
	points = np.insert(points, 0, 1, axis=1) 
	points = np.insert(points, 3, values=points[:,1]*points[:,2], axis=1) 
	points = np.insert(points, 4, values=points[:,1]**2, axis=1)
	points = np.insert(points, 5, values=points[:,2]**2, axis=1)
	return np.matrix(points)

def run_with_transformation(points, points_classification):
	points_matrix = get_transformed_points_matrix(points)

	points_pseudo_inverse = np.dot(np.linalg.inv(np.dot(points_matrix.T, points_matrix)), 
		points_matrix.T)

	return np.dot(points_pseudo_inverse, points_classification).T, points_matrix
