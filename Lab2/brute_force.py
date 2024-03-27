from itertools import product, compress
import time

from data import *


def fitness(items, knapsack_max_capacity, solution):
    total_weight = sum(compress(items['Weight'], solution))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], solution))


items, knapsack_max_capacity = get_small()
print(items)

start_time = time.time()
best_solution = None
best_value = 0
for solution in product([False, True], repeat=len(items)):
    solution_value = fitness(items, knapsack_max_capacity, solution)
    if solution_value > best_value:
        best_solution = solution
        best_value = solution_value
end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_value)
print('Time: ', total_time)
