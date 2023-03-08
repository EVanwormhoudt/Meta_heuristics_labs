import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class Colony:
    cost_matrix: np.ndarray[np.ndarray[int]]
    colony_size: int
    k_update: int
    alpha: float
    reduction_rate: float
    iterations: int
    test = 1

    best_path: list[int] = None
    best_cost: int = None
    iterations_2: list[int] = None
    pheromone_matrix: np.ndarray[np.ndarray[int]] = None
    ants: list[list[int]] = None

    def __post_init__(self):
        self.pheromone_matrix = np.ones(self.cost_matrix.shape)
        self.ants = []
        self.beta = 1 - self.alpha

    def run(self):
        self.create_colony()
        self.iterations_2 = []
        for i in range(self.iterations):
            self.evaporate_pheromone()
            self.reinforcement_pheromone()
            self.create_colony()
            self.iterations_2.append(self.best_cost)

        self.plot()


    def create_colony(self):
        for i in range(self.colony_size):
            self.ants.append(self.create_ant())

    def evaporate_pheromone(self):
        self.pheromone_matrix = self.pheromone_matrix * (1 - self.reduction_rate)

    def reinforcement_pheromone(self):
        ants_cost = list(map(self.cost_function, self.ants))
        ants_cost_sorted = sorted(ants_cost)
        best_ants = [self.ants[ants_cost.index(ants_cost_sorted[i])] for i in range(self.k_update)]
        best_ants_cost = list(map(self.cost_function, best_ants))
        for ant in best_ants:
            for i in range(len(ant) - 1):
                self.pheromone_matrix[ant[i], ant[i + 1]] += 1 / self.cost_function(ant)
            self.pheromone_matrix[ant[-1], ant[0]] += 1 / self.cost_function(ant)

        if self.best_cost is None or ants_cost_sorted[0] < self.best_cost:
            self.best_cost = ants_cost_sorted[0]
            self.best_path = self.ants[ants_cost.index(ants_cost_sorted[0])]

    def create_ant(self):
        cities = np.array(range(self.cost_matrix.shape[0]))

        ant = []
        random_city = random.choice(cities)
        ant.append(random_city)
        cities = np.delete(cities, random_city)

        for i in range(len(cities)):
            probabilities = self.get_probability(ant, cities)
            next_city = np.random.choice(len(cities), p=probabilities)
            ant.append(cities[next_city])
            cities = np.delete(cities, next_city)

        return ant

    def get_probability(self, ant: list[int], cities: np.ndarray) -> np.ndarray[float]:

        probabilities = np.zeros(len(cities))

        distances = self.normalize_distance_matrix()
        for i, city in enumerate(cities):
            probabilities[i] = self.pheromone_matrix[ant[-1], city] ** self.alpha * (
                    1 / distances[ant[-1], city]) ** self.beta

        probabilities = probabilities / np.sum(probabilities)
        return probabilities

    def normalize_distance_matrix(self):
        distances = deepcopy(self.cost_matrix)
        max_distance = np.max(distances)
        distances = distances / max_distance
        return distances

    def cost_function(self, ant: list[int]) -> int:
        cost = 0
        for i in range(len(ant) - 1):
            cost += self.cost_matrix[ant[i], ant[i + 1]]
        cost += self.cost_matrix[ant[-1], ant[0]]
        return cost

    def plot(self):
        print('Best cost: ', self.best_cost, 'with params: ', self.alpha, self.beta, self.reduction_rate, self.k_update,
              self.colony_size)
        plt.plot(self.iterations_2)
        plt.show()


def main():
    matrix = np.loadtxt("../Data/gr17.2085.tsp", dtype=int)
    ants = [50, 100, 200]
    best_ants = [5, 15, 50]
    alpha = [0.2, 0.5, 0.8]
    reduction_rate = [0.2, 0.5, 0.8]

    for i in ants:
        for j in best_ants:
            for k in alpha:
                for l in reduction_rate:
                    colony = Colony(matrix, i, j, k, l, 100)
                    colony.run()


main()
