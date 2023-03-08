import math
import random

import numpy as np
from copy import deepcopy

POP_SIZE = 100
LENGTH = 25
PARENTS_SIZE = 0.5
TOURNAMENT_SIZE = 5


def generate():
    values = []
    for _ in range(POP_SIZE):
        values.append(np.random.randint(low=0, high=2, size=LENGTH))
    return values


def score(value):
    return np.sum(value)


def remove_from_population(population, value):
    for member in range(len(population)):
        if np.equal(value, population[member]).all():
            population.pop(member)
            break
    return population


def roulette(population):
    scores = np.array(list(map(score, population)))
    total = np.sum(scores)
    proba = scores / total
    parents = np.random.choice(len(population), size=math.floor(POP_SIZE * PARENTS_SIZE), p=proba)
    parents = [population[i] for i in parents]
    return parents


def tournament(population):
    parents = []
    for i in range(math.floor(POP_SIZE * PARENTS_SIZE)):
        indexes = np.random.choice(len(population), size=TOURNAMENT_SIZE)
        competitors = [population[i] for i in indexes]
        scores = np.array(list(map(score, competitors)))
        test = np.argmax(scores)
        parents.append(competitors[np.argmax(scores)])

    return parents


def rank(population):
    pressure = 1.5
    scores = np.array(list(map(score, population)))
    indexes = np.argsort(scores)
    proba = (2-pressure)/len(population) + 2*(indexes)*(pressure-1)/(len(population)*(len(population)-1))
    parents = np.random.choice(len(population), size=math.floor(POP_SIZE * PARENTS_SIZE), p=proba)
    parents = [population[i] for i in parents]
    return parents


def selection(population, method: str):
    match method:
        case "roulette":
            return roulette(population)
        case "tournament":
            return tournament(population)
        case "rank":
            return rank(population)
        case _:
            raise Exception("Invalid method")


def crossover(parents):
    children = []
    parents_copy = deepcopy(parents)

    while len(parents_copy):
        parent1 = parents_copy.pop()
        parent2 = parents_copy.pop()
        crossover_point = random.randint(0, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        children.append(child1)
        children.append(child2)
    return children


def mutation(children):
    for i in range(len(children)):
        if np.random.rand() < 0.1:
            children[i][np.random.randint(0, LENGTH)] = 1 - children[i][np.random.randint(0, LENGTH)]
    return children


def generational(population, children, parents):
    for parent in parents:
        remove_from_population(population, parent)

    for child in children:
        population.append(child)

    return population


def steady_state(population, children, parents):
    for i in range(len(parents) // 2):
        parent1 = parents.pop()
        parent2 = parents.pop()
        child1 = children.pop()
        child2 = children.pop()

        score_parent1 = score(parent1)
        score_parent2 = score(parent2)
        score_child1 = score(child1)
        score_child2 = score(child2)

        if score_parent1 > score_parent2:
            remove_from_population(population, parent1)
            if score_child1 > score_child2:
                population.append(child1)
            else:
                population.append(child2)
        else:
            remove_from_population(population, parent2)
            if score_child1 > score_child2:
                population.append(child1)
            else:
                population.append(child2)
    return population


def elitism(population, children, parents):
    population = np.array(population)
    children = np.array(children)
    parents = np.array(parents)
    population = np.concatenate([population, np.array(children), np.array(parents)], axis=0)
    scores = np.array(list(map(score, population)))

    population = population[np.argsort(scores)]
    population = population[::-1]
    population = population[:POP_SIZE]
    return population


def replacement(population, parents, children, method: str):
    match method:
        case "generational":
            return generational(population, children, parents)
        case "steady-state":
            return steady_state(population, children, parents)
        case "elitism":
            return elitism(population, children, parents)
        case _:
            raise Exception("Invalid method")


def genetic_algorithm(selection_method: str, replacement_method: str):
    population = generate()
    max_individual = max(list(map(score, population)))
    i = 0
    while max_individual < 25:
        parents = selection(population, selection_method)
        children = crossover(parents)
        children = mutation(children)
        population = replacement(population, parents, children, replacement_method)
        max_individual = max(list(map(score, population)))
        i += 1
        print("the max of the population",i," is: ", max_individual)


    scores = np.array(list(map(score, population)))
    return i


# for selection_method in ["roulette", "tournament", "rank"]:
#     for replacement_method in ["generational", "steady-state", "elitism"]:
#         scores = []
#         for i in range(100):
#             scores.append(genetic_algorithm(selection_method, replacement_method))
#         print(f"{selection_method} {replacement_method} {np.mean(scores)}")

genetic_algorithm("tournament", "elitism")
