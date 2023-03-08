import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

matrix = np.loadtxt("../Data/gr17.2085.tsp")

SIZE = 17
initial_state = list(range(SIZE))
random.shuffle(initial_state)
iterations = []


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


def tabu(initial_state):
    state = initial_state

    cost_state = cost_function(state)
    tabu_list = []
    max_iter = 1000
    best_cost = cost_state
    best_state = state
    while max_iter > 0:
        neighbours = create_neighbours(state)
        costs = list(map(cost_function, neighbours))
        for i in range(len(costs)):
            local_best = min(costs)
            local_best_state = neighbours[costs.index(local_best)]
            if local_best_state in tabu_list:
                costs.remove(local_best)
            else:
                break
        cost_state = min(costs)
        if cost_state < best_cost:
            best_state = state
            best_cost = cost_state
        state = neighbours[costs.index(cost_state)]

        tabu_list.append(state)
        iterations.append(cost_state)
        max_iter -= 1

    return best_cost, best_state


print("best path : " + str(tabu(initial_state)))

sns.lineplot(x=np.arange(len(iterations)), y=np.array(iterations))
plt.show()
