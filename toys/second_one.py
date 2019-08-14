from common.common_functions import get_target_function, \
	get_uniform_random_points_classification, \
	get_uniform_random_points_classification_with_predefined_target_function
import numpy as np

from algorithms import linear_regression, perceptron_learning_algorithm


def get_coin_min_index(heads_result, tosses):
	min_heads = tosses
	coin_min_index = 0
	for i, x in enumerate(heads_result):
		if x < min_heads:
			coin_min_index = i
			min_heads = x
	return coin_min_index


def calculate_coin_fraction(coin_index, heads_result, tosses):
	return heads_result[coin_index] / tosses


def run_experiment(coins_quantity, tosses):
	head = 1
	coin1_index = 0
	coin_rand_index = np.random.randint(0, coins_quantity)
	coin_min_index = None

	flip_coins_data = np.random.randint(head-1, head+1, 
		size=(coins_quantity, tosses))
	heads_result = np.sum(flip_coins_data, axis=1)

	coin_min_index = get_coin_min_index(heads_result, tosses)

	coin1_fraction = calculate_coin_fraction(coin1_index, heads_result, tosses)
	coin_rand_fraction = calculate_coin_fraction(coin_rand_index, heads_result, 
		tosses)
	coin_min_fraction = calculate_coin_fraction(coin_min_index, heads_result, 
		tosses)

	return coin1_fraction, coin_rand_fraction, coin_min_fraction


def hoeffding_inequality_part(run_times=100000, coins_quantity=1000, tosses=10):
	average_fractions = list()
	for i in range(run_times):
		coin1_fraction, coin_rand_fraction, coin_min_fraction = run_experiment(
			coins_quantity, tosses)

		average_fractions.append([coin1_fraction, coin_rand_fraction, 
			coin_min_fraction])

	print('Average value of Cmin fraction: {}'.format(np.mean([i[2] for i in average_fractions])))
	print('Average value of C1 fraction: {}'.format(np.mean([i[0] for i in average_fractions])))
	print('Average value of Crand fraction: {}'.format(np.mean([i[1] for i in average_fractions])))


def get_out_of_sample_errors(points, target_functions_list, up_line_classification_list, weights_list):
	out_of_sample_errors = list()
	points_matrix = linear_regression.get_points_adapted_matrix(points)
	for i, weights in enumerate(weights_list):
		points_classification = \
			get_uniform_random_points_classification(points, 
				target_functions_list[i], up_line_classification_list[i])

		out_linear_regression_classification = \
			linear_regression.classify_points_given_weights(points_matrix, 
				weights)

		out_of_sample_error = \
			sum([1 for i, x in enumerate(points_classification) 
				if x != out_linear_regression_classification[i]]) / \
					len(points_classification)

		out_of_sample_errors.append(out_of_sample_error)
	return out_of_sample_errors


def linear_regression_part1(n_points, run_times=1000, y_min=-1, y_max=1):
	in_sample_errors = list()
	weights_list = list()
	target_functions_list = list()
	up_line_classification_list = list()

	for i in range(run_times):
		target_function = get_target_function()
		points = np.random.uniform(y_min, y_max, size=(n_points, 2))
		up_line_classification = np.random.choice([y_max, y_min])

		points_classification = get_uniform_random_points_classification(
			points, target_function, up_line_classification)

		weights, points_matrix = linear_regression.run(points, points_classification)

		linear_regression_classification = \
			linear_regression.classify_points_given_weights(points_matrix, 
				weights)

		in_sample_error = sum([1 for i, x in enumerate(points_classification) 
			if x != linear_regression_classification[i]]) / n_points

		output_str = f'iteration: {i}, E_in: {in_sample_error}'
		print(output_str, end='\r'*len(output_str))

		target_functions_list.append(target_function)
		up_line_classification_list.append(up_line_classification)
		weights_list.append(weights)
		in_sample_errors.append(in_sample_error)

	print('Average in sample error: {}'.format(np.mean(in_sample_errors)))

	fresh_points = np.random.uniform(y_min, y_max, size=(1000, 2))

	out_of_sample_errors = get_out_of_sample_errors(fresh_points, 
		target_functions_list, up_line_classification_list, weights_list)

	print('Average out of sample error: {}'.format(np.mean(out_of_sample_errors)))


def linear_regression_part2(n_points, run_times=1000, y_min=-1, y_max=1):
	iterations_list = list()

	for i in range(run_times):
		target_function = get_target_function()
		points = np.random.uniform(y_min, y_max, size=(n_points, 2))
		up_line_classification = np.random.choice([y_max, y_min])

		points_classification = get_uniform_random_points_classification(
			points, target_function, up_line_classification)

		weights, _ = linear_regression.run(points, points_classification)

		_, iterations = perceptron_learning_algorithm.run(
			points, points_classification, np.ravel(weights))

		iterations_list.append(iterations)

	print(f'Average iterations: {np.mean(iterations_list)}')


def add_noise_to_points_classification(points_classification, n_points, 
	percentage=0.1):
	points_classification_copy = points_classification[:]
	for index in range(0, int(n_points*percentage)):
		points_classification_copy[index] *= -1
	return points_classification_copy


def calculate_avg_E_in_adding_noise_to_data(n_points, run_times, y_min, y_max):
	in_sample_errors = list()
	for i in range(run_times):
		points = np.random.uniform(y_min, y_max, size=(n_points, 2))

		points_classification = \
		get_uniform_random_points_classification_with_predefined_target_function(
			points)

		noisy_points_classification = add_noise_to_points_classification(
			points_classification, n_points)

		weights, points_matrix = linear_regression.run(points, 
			noisy_points_classification)

		linear_regression_classification = \
			linear_regression.classify_points_given_weights(points_matrix, 
				weights)

		in_sample_error = sum([1 for i, x in enumerate(noisy_points_classification) 
			if x != linear_regression_classification[i]]) / n_points

		output_str = f'iteration: {i}, E_in: {in_sample_error}'
		print(output_str, end='\r'*len(output_str))

		in_sample_errors.append(in_sample_error)

	return in_sample_errors


def calculate_avg_E_out_adding_noise_to_data(n_points, run_times, y_min, y_max, weights):
	out_sample_errors = list()
	for i in range(run_times):
		points = np.random.uniform(y_min, y_max, size=(n_points, 2))

		points_classification = \
		get_uniform_random_points_classification_with_predefined_target_function(
			points)

		noisy_points_classification = add_noise_to_points_classification(
			points_classification, n_points)

		points_matrix = linear_regression.get_transformed_points_matrix(points)

		linear_regression_classification = \
			linear_regression.classify_points_given_weights(points_matrix, 
				weights).T

		out_sample_error = sum([1 for i, x in enumerate(noisy_points_classification) 
			if x != linear_regression_classification[i]]) / n_points

		output_str = f'iteration: {i}, E_out: {out_sample_error}'
		print(output_str, end='\r'*len(output_str))

		out_sample_errors.append(out_sample_error)

	return out_sample_errors


def calculate_average_weights_when_transforming_data(n_points, run_times, 
	y_min, y_max):
	weights_list = list()

	for i in range(run_times):
		points = np.random.uniform(y_min, y_max, size=(n_points, 2))

		points_classification = \
		get_uniform_random_points_classification_with_predefined_target_function(
			points)

		weights, _ = linear_regression.run_with_transformation(points, 
			points_classification)
		weights_list.append(np.ravel(weights))

	return [np.mean([weights[index] for weights in weights_list]) 
		for index in range(len(weights))]


def nonlinear_regression_part(n_points=1000, run_times=1000, 
	y_min=-1, y_max=1):

	in_sample_errors = \
		calculate_avg_E_in_adding_noise_to_data(n_points, run_times, y_min, y_max)

	print('Average E_in when data is noisy: {}'.format(np.mean(in_sample_errors)))

	average_weights = calculate_average_weights_when_transforming_data(n_points, 
		run_times, y_min, y_max)
	print(f'Average weights when data is transformed: {average_weights}')

	weights = np.array(average_weights)

	out_of_sample_errors = \
		calculate_avg_E_out_adding_noise_to_data(n_points, run_times, y_min, y_max, weights)
	print('Average E_out when data is noisy: {}'.format(np.mean(out_of_sample_errors)))


if __name__ == '__main__':
	# hoeffding_inequality_part()
	# linear_regression_part1(n_points=100)
	# linear_regression_part2(n_points=10)
	nonlinear_regression_part(n_points=1000)
