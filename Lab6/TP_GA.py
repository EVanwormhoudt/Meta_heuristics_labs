# Genetic algorithm for the max one problem
import random
import numpy as np
import matplotlib.pyplot as plt




def generate_initial_population(max_one_size):
    population = []
    for i in range(population_size // 3):
        individual = []
        for j in range(max_one_size):
            individual.append(random.randint(0, 1))
        population.append(individual)

    while len(population) < population_size:
        individual = []
        for j in range(max_one_size):
            individual.append(random.randint(0, 1))
        if individual not in population:
            population.append(individual)
    return population  # return a list of possible solutions


def evaluate_population(population, best_fitnesses, best_solutions):
    evaluated_population = []
    best_solution = population[0]
    best_fitness = 0
    for individual in population:
        evaluated_population.append((individual, np.sum(individual)))
    for solution in evaluated_population:
        if solution[1] > best_fitness:
            best_fitness = solution[1]
            best_solution = solution[0]
    best_fitnesses.append(best_fitness)
    best_solutions.append(best_solution)
    return evaluated_population  # return a list of tuples (solution, fitness)


def tournament_selection(evaluated_population, tournament_size):
    parents_population = []
    tournament = []
    population = evaluated_population.copy()
    for i in range(population_size // 2):
        for j in range(tournament_size):
            tournament.append(random.choice(population))
        tournament.sort(key=lambda x: x[1], reverse=True)
        parents_population.append(tournament[0])
        population.remove(tournament[0])
        tournament.clear()

    return parents_population


def roulette_wheel_selection(evaluated_population):
    parents_population = []
    fitness_sum = 0
    for individual in evaluated_population:
        fitness_sum += individual[1]
    parents_population.append(np.random.choice(evaluated_population, size=population_size // 2, replace=False,
                                               p=[individual[1] / fitness_sum for individual in
                                                  evaluated_population]))
    return parents_population


def rank_based_selection(evaluated_population):
    population = evaluated_population.copy()
    population.sort(key=lambda x: x[1], reverse=False)
    probability = []
    mu = population_size
    selection_pressure = 1.5
    for i in range(population_size):
        probability.append((2 - selection_pressure) / mu + 2 * i * (selection_pressure - 1) / (mu * (mu - 1)))
    parents_population = np.random.choice(evaluated_population, size=population_size // 2, replace=False, p=probability)
    return parents_population


def selection(evaluated_population, method):
    tournament_size = 5
    parents_population = []
    if method == 'roulette':
        # Roulette wheel selection
        parents_population = roulette_wheel_selection(evaluated_population)

    elif method == 'tournament':
        # Tournament selection
        parents_population = tournament_selection(evaluated_population, tournament_size)

    elif method == 'rank':
        # Rank selection
        parents_population = rank_based_selection(evaluated_population)

    else:
        print('Error: no selection method selected')
    return parents_population


def crossover(parent1, parent2):
    child1 = []
    child2 = []
    crossover_point = random.randint(0, len(parent1) - 1)
    for i in range(len(parent1)):
        if i < crossover_point:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2


def mutation(child):
    mutation_point = random.randint(0, len(child) - 1)
    if child[mutation_point] == 0:
        child[mutation_point] = 1
    else:
        child[mutation_point] = 0


def reproduction(parents_population):
    children_population = []
    prob_crossover = 0.90
    prob_mutation = 0.1
    for i in range(0, population_size // 2, 2):
        parent1 = parents_population[i][0]
        parent2 = parents_population[i + 1][0]
        crossover_prob = np.random.choice([True, False], 1, p=[prob_crossover, 1 - prob_crossover])
        if crossover_prob:
            child1, child2 = crossover(parent1, parent2)
            children_population.append(child1)
            children_population.append(child2)
        else:
            children_population.append(parent1)
            children_population.append(parent2)

    for child in children_population:
        mutation_prob = np.random.choice([True, False], 1, p=[prob_mutation, 1 - prob_mutation])
        if mutation_prob:
            mutation(child)
    return children_population


def elitism(parents_population, children_population, population):
    new_population = []
    new_population.extend(parents_population)
    new_population.extend(children_population)
    new_population.sort(key=lambda x: x[1], reverse=True)
    population_2 = [person for person in population if person not in parents_population]
    new_population = new_population[:population_size // 2]
    new_population.extend(population_2)
    return new_population


def generational(parents_population, children_population, population):
    new_population = []
    population_2 = [person for person in population if person not in parents_population]
    new_population.extend(children_population.copy())
    new_population.extend(population_2)
    return new_population


def steady_state(parents_population, children_population, population):
    new_population = []
    population_2 = [person for person in population if person not in parents_population]
    for i in range(0, population_size // 2, 2):
        if children_population[i][1] > children_population[i + 1][1]:
            if parents_population[i][1] > parents_population[i][1]:
                new_population.append(parents_population[i])
                new_population.append(children_population[i])
            else:
                new_population.append(children_population[i])
                new_population.append(parents_population[i + 1])
        else:
            if parents_population[i][1] > parents_population[i][1]:
                new_population.append(parents_population[i])
                new_population.append(children_population[i + 1])
            else:
                new_population.append(parents_population[i + 1])
                new_population.append(children_population[i + 1])
    new_population.extend(population_2)
    return new_population


def replacement(parents_population, children_population, method, population):
    new_population = []
    if method == 'elitism':
        new_population = elitism(parents_population, children_population, population)
    elif method == 'generational':
        new_population = generational(parents_population, children_population, population)
    elif method == 'steady_state':
        new_population = steady_state(parents_population, children_population, population)
    else:
        print('Error: no replacement method selected')
    return new_population


def genetic_algorithm(max_one_size):
    best_fitnesses = []
    best_solutions = []
    initial_population = generate_initial_population(max_one_size)
    evaluated_population = evaluate_population(initial_population, best_fitnesses, best_solutions)
    t = 0
    while t < 100:
        select_parents = selection(evaluated_population, "roulette")
        children_population = reproduction(select_parents)
        evaluated_children = evaluate_population(children_population, best_fitnesses, best_solutions)
        evaluated_population = replacement(select_parents, evaluated_children, "elitism", evaluated_population)
        t += 1
        print("Generation : ", t, "best fitness : ", best_fitnesses[t-1], "best solution : ", best_solutions[t-1])
    return best_fitnesses, best_solutions


if __name__ == '__main__':
    population_size = 100
    best_fitnesses, best_solutions = genetic_algorithm(25)
    plt.plot(best_fitnesses)
    plt.show()
    best_fitness = max(best_fitnesses)
    best_solution = best_solutions[best_fitnesses.index(best_fitness)]
    print('Best fitness: ', best_fitness, 'Best solution: ', best_solution)
