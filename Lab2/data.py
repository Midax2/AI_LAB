import pandas as pd

def get_small():
    knapsack = pd.read_csv('knapsack-small.csv')
    return knapsack, 10

def get_big():
    knapsack = pd.read_csv('knapsack-big.csv')
    return knapsack, 6404180
