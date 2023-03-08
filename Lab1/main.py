import numpy as np

from time import time

with open("../Data/p01.15.291.tsp", "r") as f:
    lines = f.read().splitlines()

matrix = [[int(x) for x in line.split("        ")] for line in lines]

matrix = np.array(matrix)

LENGHT = 5


def find_routes(matrix):
    routes = []
    queue = []
    queue.append([0])
    while queue:
        route = queue.pop(0)
        if len(route) == len(matrix):
            routes.append(route)
            continue
        else:
            for i in range(len(matrix)):
                if i not in route:
                    queue.append(route + [i])
    print(len(routes))
    return routes


def find_min_distance(routes, matrix):
    distances = []

    for route in routes:
        distance = 0
        for i in range(len(route) - 1):
            distance += matrix[route[i]][route[i + 1]]
        distance += matrix[route[-1]][route[0]]
        distances.append(distance)

    shortest = min(distances)
    path = np.array(routes[distances.index(min(distances))]) + 1
    return shortest, path


matrix = matrix[:LENGHT, :LENGHT]

start = time()
print(matrix)
# print(find_min_distance(find_routes(matrix), matrix))

import itertools

array = itertools.permutations(range(0, 5))

new_array = [list(value) for value in array]

print(find_min_distance(new_array, matrix))

print(time() - start)
