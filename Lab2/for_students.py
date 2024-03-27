from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_wheel_selection(items, knapsack_max_capacity, population, n_selection):
    selected = []
    fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    total_fitness = sum(fitnesses)
    probabilities = [fit / total_fitness for fit in fitnesses]
    for i in range(0, n_selection):
        selected_index = random.choices(range(len(population)), probabilities)[0]
        selected.append(population[selected_index])
    return selected


def crossover(population, population_size, n_selection):
    new_population = []
    while len(new_population) < population_size:
        random_parent_1 = random.randint(0, n_selection - 1)
        random_parent_2 = random_parent_1
        while random_parent_2 == random_parent_1:
            random_parent_2 = random.randint(0, n_selection - 1)
        parent1 = population[random_parent_1]
        parent2 = population[random_parent_2]
        crossover_point = len(parent1) // 2
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        new_population.append(child1)
        new_population.append(child2)
    return new_population

def mutation(population):
    new_population = []
    for individual in population:
        temp = individual
        rand_bit_id = random.randint(0, len(individual) - 1)
        temp[rand_bit_id] = ~temp[rand_bit_id]
        new_population.append(temp)
    return new_population

items, knapsack_max_capacity = get_big()

population_size = 100
generations = 200
n_elite = 1
n_selection = 20

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # Selekcja
    selected = roulette_wheel_selection(items, knapsack_max_capacity, population, n_selection)

    # Elityzm
    elite = sorted(population, key=lambda x: fitness(items, knapsack_max_capacity, x), reverse=True)[:n_elite]

    selected += elite

    # Krzyżowanie
    children = crossover(selected, population_size, n_selection + n_elite)

    # Mutacja
    mutants = mutation(children)

    # Nowa populacja
    population = mutants

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()