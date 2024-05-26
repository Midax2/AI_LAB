import pandas as pandas
import numpy as np
import random
from deap import base, creator, tools, algorithms

data = pandas.read_table('Words.txt')
import random

def can_place_word(board, word, row, col, direction):
    if direction == 'H':
        if col + len(word) > len(board[0]):
            return False
        for i in range(len(word)):
            if board[row][col + i] not in (' ', word[i]):
                return False
    elif direction == 'V':
        if row + len(word) > len(board):
            return False
        for i in range(len(word)):
            if board[row + i][col] not in (' ', word[i]):
                return False
    return True

def place_word(board, word, row, col, direction):
    if direction == 'H':
        for i in range(len(word)):
            board[row][col + i] = word[i]
    elif direction == 'V':
        for i in range(len(word)):
            board[row + i][col] = word[i]

def is_valid_board(board, words):
    valid_words = set(words)
    for row in board:
        row_words = ''.join(row).split()
        for rw in row_words:
            if rw and rw not in valid_words:
                return False

    for col in range(len(board[0])):
        col_words = ''.join(board[row][col] for row in range(len(board))).split()
        for cw in col_words:
            if cw and cw not in valid_words:
                return False

    return True

class Individual:
    def __init__(self, words, board_size):
        self.words = words
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.fitness = None
        self.place_words()

    def place_words(self):
        for word in self.words:
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                attempts += 1
                row = random.randint(0, self.board_size - 1)
                col = random.randint(0, self.board_size - 1)
                direction = random.choice(['H', 'V'])
                if can_place_word(self.board, word, row, col, direction):
                    place_word(self.board, word, row, col, direction)
                    placed = True
        if not is_valid_board(self.board, self.words):
            self.fitness = float('inf')
        else:
            self.calculate_fitness()

    def calculate_fitness(self):
        min_row, max_row, min_col, max_col = self.get_bounds()
        if min_row == self.board_size:
            self.fitness = float('inf')
        else:
            self.fitness = (max_row - min_row + 1) * (max_col - min_col + 1)

    def get_bounds(self):
        min_row = self.board_size
        max_row = 0
        min_col = self.board_size
        max_col = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != ' ':
                    if row < min_row:
                        min_row = row
                    if row > max_row:
                        max_row = row
                    if col < min_col:
                        min_col = col
                    if col > max_col:
                        max_col = col
        return min_row, max_row, min_col, max_col


def genetic_algorithm(words, board_size, population_size, generations, mutation_rate):
    population = [Individual(words, board_size) for _ in range(population_size)]

    for generation in range(generations):
        population.sort(key=lambda x: x.fitness)
        if generation % 10 == 0:
            print(f"Generation {generation} Best fitness: {population[0].fitness}")

        next_population = population[:population_size // 2]

        while len(next_population) < population_size:
            parent1, parent2 = random.sample(population[:population_size // 2], 2)
            child1, child2 = crossover(parent1, parent2, board_size)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            child1.calculate_fitness()
            child2.calculate_fitness()
            next_population.extend([child1, child2])

        population = next_population[:population_size]

    population.sort(key=lambda x: x.fitness)
    return population[0]

def crossover(parent1, parent2, board_size):
    crossover_point = random.randint(1, len(parent1.words) - 1)
    child1_words = parent1.words[:crossover_point] + parent2.words[crossover_point:]
    child2_words = parent2.words[:crossover_point] + parent1.words[crossover_point:]
    child1 = Individual(child1_words, board_size)
    child2 = Individual(child2_words, board_size)
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual.words)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual.words) - 1)
            individual.words[i], individual.words[j] = individual.words[j], individual.words[i]


def print_board(board):
    for row in board:
        print(' '.join(row))


words = []
for word in data['Words'].values:
    words.append(str.upper(word))
best_solution = genetic_algorithm(words, board_size=20, population_size=30, generations=100, mutation_rate=0.1)
print_board(best_solution.board)
