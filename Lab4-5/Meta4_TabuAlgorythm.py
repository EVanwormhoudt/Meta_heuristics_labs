import numpy as np
import itertools
import random
import copy
import math

#input = np.loadtxt("C:/Users/Ciri/Desktop/Навчання/France/p01.15.291.tsp", dtype='i')
input = np.loadtxt("../Data/p01.15.291.tsp")

cities = list(range(15))
random.shuffle(cities)
tabu = []

print('\n\n', cities, '\n\n')


def get_neighbors(state):
    neighbors = []
    for swap1 in range(0, len(state)):
        for swap2 in range(0, len(state)):
            state_copy = copy.deepcopy(state)
            tmp = state_copy[swap1]
            state_copy[swap1] = state_copy[swap2]
            state_copy[swap2] = tmp
            if state_copy not in neighbors and state_copy != state:
                neighbors.append(state_copy)
    return neighbors


def objective_function(state):
    length = 0
    for i in range(0, len(state)):
        length += input[state[i], state[(i + 1) % len(state)]]
    return length


iterations = 0
global_best_length = 100000
global_best_path = cities

local_best_path = cities

while iterations < 500:
    iterations += 1
    all_path = get_neighbors(local_best_path)

    all_path_lengths = list(map(objective_function, all_path))

    for i in range(len(all_path_lengths)):
        local_best_length = min(all_path_lengths)
        if all_path[all_path_lengths.index(local_best_length)] in tabu:
            all_path_lengths.remove(local_best_length)
        else:
            break

    local_best_path = all_path[all_path_lengths.index(local_best_length)]

    if local_best_length < global_best_length:
        global_best_path = local_best_path
        global_best_length = local_best_length

    tabu.append(local_best_path)
    cities = local_best_path
    print( 'Path: ', local_best_path, '\nLength: ', local_best_length, '\n\n')

print("=========================================")
print('\n\nBest path: ', global_best_path, '\nBest length: ', global_best_length, '\n\n')
