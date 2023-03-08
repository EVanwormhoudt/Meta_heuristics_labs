import numpy as np
from itertools import permutations
from time import time
import random
from copy import deepcopy

with open("../Data/p01.15.291.tsp", "r") as f:
    lines = f.read().splitlines()

matrix = [[int(x) for x in line.split("        ")] for line in lines]

matrix = np.array(matrix)
matrix = np.loadtxt("../Data/br17.39 (1).atsp")

SIZE = 17
initial_state = list(range(SIZE))
random.shuffle(initial_state)


def cost_function(x):
    return np.sum(np.fromiter((matrix[x[i], x[i + 1]] for i in range(len(x) - 1)), dtype=float)) + matrix[x[-1], x[0]]


def create_neighbours(path):
    permutations = []
    for i in range(SIZE):
        for j in range(i + 1, SIZE):
            permutations.append((i, j))

    paths = []
    for permutation in permutations:
        path_copy = deepcopy(path)
        tmp = path_copy[permutation[0]]
        path_copy[permutation[0]] = path_copy[permutation[1]]
        path_copy[permutation[1]] = tmp
        paths.append(path_copy)
    return paths


def best_improvment(initial_state):
    state = initial_state

    cost_state = cost_function(state)

    while True:
        neighbours = create_neighbours(state)
        costs = list(map(cost_function, neighbours))
        better_costs = list(filter(lambda x: x < cost_state, costs))

        if not better_costs:
            break

        cost_state = min(better_costs)
        state = neighbours[costs.index(cost_state)]

    return cost_state


count = 0
while True:
    count = 0
    for i in range(1000):
        random.shuffle(initial_state)
        if best_improvment(initial_state) == 39:
            count += 1
    print(count)

print(count / 10)
