import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

matrix = np.loadtxt("../Data/gr17.2085.tsp")

SIZE = 17
Matrix = np.ndarray[np.ndarray[int]]
Path = list[int]
Path_and_cities = tuple[Path, list[int]]

initial_state = list(range(SIZE))
random.shuffle(initial_state)
iterations = []


def cost_function(x: Path):
    return np.sum(np.fromiter((matrix[x[i], x[i + 1]] for i in range(len(x) - 1)), dtype=float)) + matrix[x[-1], x[0]]


def update_memory(long_term_memory: Matrix, medium_term_memory: Matrix, path: Path):
    for i in range(len(path)):
        long_term_memory[path[i], i] += 1

    for i in range(len(path)):
        if medium_term_memory[path[i], i] > 0:
            medium_term_memory[path[i], i] += 1
        else:
            medium_term_memory[:, i] = np.zeros((1, SIZE))
            medium_term_memory[path[i], i] += 1


def intensification(medium_term_memory: Matrix, path: Path) -> list[int]:
    blocked = []
    # add 1/4 of the most visited city in each column to the blocked list
    medium_term_memory_copy = deepcopy(medium_term_memory)

    for i in range(5):
        value = np.argmax(medium_term_memory_copy)
        medium_term_memory_copy[value // SIZE, value % SIZE] = 0

        blocked.append(value // SIZE)

    return blocked


def diversification(long_term_memory: Matrix, path: Path) -> Path:
    new_path = []

    long_term_memory_copy = deepcopy(long_term_memory)
    for i in range(len(path)):
        value = np.argmin(long_term_memory_copy[:, i])
        if isinstance(value, np.int64):
            new_path.append(value)
        else:
            new_path.append(value[0])
        long_term_memory_copy[value, :] = np.zeros((1, SIZE)) + 9999

    return new_path


def create_neighbours(path: Path, blocked: list[int] = None) -> list[Path_and_cities]:
    permutations = []
    if blocked is None:
        blocked = []
    permutations = [(i, j) for i in range(SIZE) for j in range(i + 1, SIZE) if i not in blocked and j not in blocked]

    paths = []
    for permutation in permutations:
        path_copy = deepcopy(path)
        tmp = path_copy[permutation[0]]
        path_copy[permutation[0]] = path_copy[permutation[1]]
        path_copy[permutation[1]] = tmp
        paths.append((path_copy, list(permutation)))
    return paths


def tabu(initial_state):
    state = initial_state

    time_since_improvement = 0
    cost_state = cost_function(state)
    tabu_list = []

    best_cost = cost_state
    best_state = state
    MAX_ITER = 1000
    blocked = []

    medium_term_memory = np.zeros((SIZE, SIZE))
    long_term_memory = np.zeros((SIZE, SIZE))

    while MAX_ITER > 0:
        neighbour_list = create_neighbours(state, blocked)
        neighbours = [neighbour[0] for neighbour in neighbour_list]
        permutations = [neighbour[1] for neighbour in neighbour_list]
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
            time_since_improvement = 0
        else:
            time_since_improvement += 1

        state = neighbours[costs.index(cost_state)]

        tabu_list.append(state)
        update_memory(long_term_memory, medium_term_memory, state)

        if len(tabu_list) > 10:
            tabu_list.pop(0)

        if time_since_improvement > 20 and len(blocked) == 0:
            blocked = intensification(medium_term_memory, state)

        if time_since_improvement > 50:
            state = diversification(long_term_memory, state)
            blocked = []
            time_since_improvement = 0

        iterations.append(cost_state)
        MAX_ITER -= 1

    return best_cost, best_state


print("best path : " + str(tabu(initial_state)))

sns.lineplot(x=np.arange(len(iterations)), y=np.array(iterations))
plt.show()
