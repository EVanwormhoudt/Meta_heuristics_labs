import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns


matrix = np.loadtxt("../Data/p01.15.291.tsp")

SIZE = 15
initial_state = list(range(SIZE))
random.shuffle(initial_state)
iterations = []


def cost_function(x):
    return np.sum(np.fromiter((matrix[x[i], x[i + 1]] for i in range(len(x) - 1)), dtype=float)) + matrix[x[-1], x[0]]


def create_random_neighbour(path):
    i = random.randint(0, SIZE - 1)
    j = random.randint(0, SIZE - 1)
    while j == i:
        j = random.randint(0, SIZE - 1)

    path_copy = deepcopy(path)
    tmp = path_copy[i]
    path_copy[i] = path_copy[j]
    path_copy[j] = tmp
    return path_copy


def simulated_annealing(inititial_state, initial_temperature,min_temperature, cooling_rate, max_iterations):
    s = inititial_state
    best_s = s
    temperature = initial_temperature
    while temperature > min_temperature:
        for i in range(max_iterations):
            s_prime = create_random_neighbour(s)
            delta = cost_function(s_prime) - cost_function(s)
            if delta < 0:
                s = s_prime
                if cost_function(s) < cost_function(best_s):
                    best_s = s
            else:
                if random.random() < np.exp(-delta/temperature):
                    s = s_prime
        iterations.append(cost_function(s))
        temperature = temperature * cooling_rate
    return best_s


print(cost_function(simulated_annealing(initial_state, 100, 1, 0.99, 100)))
sns.lineplot(x=np.arange(len(iterations)), y=np.array(iterations))
plt.show()
