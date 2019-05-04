import numpy as np

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




if __name__ == '__main__':
	hoeffding_inequality_part()