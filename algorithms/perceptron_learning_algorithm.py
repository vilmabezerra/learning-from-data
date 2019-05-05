from numpy import random, sign, insert


def get_model_classification(weights, point):
	return sign(weights[0] + weights[1]*point[0] + weights[2]*point[1])


def get_misclassified_points(points, points_classification, weights):
	misclassified_points = list()
	for index, point in enumerate(points):
		model_classification = get_model_classification(weights, point)
		if model_classification != points_classification[index]:
			misclassified_points.append((insert(point, 0, 1, axis=0), index))
	return misclassified_points


def get_random_points_given_weights(weights, points_quantity=2):
	random_points = list()
	for i in range(points_quantity):
		random_x2 = random.uniform(size=1)[0]
		random_x1 = (- weights[0] - weights[2]*random_x2) / weights[1]
		random_points.append((random_x1, random_x2))
	return random_points


def run(points, points_classification, weights=random.uniform(size=3)):
	misclassified_points = get_misclassified_points(
			points, points_classification, weights)
	iterations = 0

	while len(misclassified_points) > 0:
		random_index = int(random.uniform(0, len(misclassified_points) - 1, size=1)[0])
		weights += misclassified_points[random_index][0] * \
			points_classification[misclassified_points[random_index][1]]

		misclassified_points = get_misclassified_points(
			points, points_classification, weights)

		iterations += 1


	return weights, iterations

def run_several_times(points, points_classification, times=1000):
	each_time_iterations = list()
	each_time_weights = list()

	for i in range(times):
		weights, iterations = run(points, points_classification)
		each_time_weights.append(weights)
		each_time_iterations.append(iterations)

	return each_time_weights, each_time_iterations
