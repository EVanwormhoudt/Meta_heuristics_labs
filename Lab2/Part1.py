import random

import numpy as np
from copy import deepcopy


initial_state = list(np.random.randint(size=5,low=0,high=2))
initial_state = [1,0,0,0,1]


def cost_function(state):
    x = sum(list([e*2**(len(state)-i-1) for i,e in enumerate(state)]))
    return x**3 - 60 * x**2 + 900 * x


def create_neighbours(state):
    neighbours = []
    state = list(state)
    for i in range(5):
        state_copy = deepcopy(state)
        state_copy[i] = (state_copy[i]+1)%2
        neighbours.append(tuple(state_copy))
    return neighbours


def get_random_element(states):
    return states[np.random.randint(size=5,low=0,high=len(states))[0]]


def print_step(neighbours_cost,i,best_state,best_cost):
    print("===================")
    print("Step " +str(i))
    for cost,neighbour in neighbours_cost.items():
        print(neighbour,str(cost))


def random_improvment(initial_state):

    state = tuple(initial_state)

    cost_state = cost_function(state)

    i = 0
    while True:
        neighbours = create_neighbours(state)
        neighbours_cost = dict(zip(map(cost_function,neighbours),neighbours))
        better_cost = list(filter(lambda x : x > cost_state,neighbours_cost.keys()))

        if not better_cost:
            break
        state = neighbours_cost[get_random_element(better_cost)]
        cost_state = cost_function(state)
        print_step(neighbours_cost,i,state,cost_state)
        i+=1

    print("Etat "+ str(state))
    print("Cost : " + str(cost_state))

def best_improvment(initial_state):
    state = tuple(initial_state)

    cost_state = cost_function(state)

    i = 0
    while True:
        neighbours = create_neighbours(state)

        neighbours_cost = dict(zip(map(cost_function, neighbours), neighbours))
        better_cost = list(filter(lambda x: x > cost_state, neighbours_cost.keys()))

        if not better_cost:
            break
        state = neighbours_cost[max(better_cost)]
        cost_state = cost_function(state)
        print_step(neighbours_cost, i, state, cost_state)
        i += 1

    print("Etat " + str(state))
    print("Cost : " + str(cost_state))

def first_improvment(initial_state):

    state = tuple(initial_state)

    cost_state = cost_function(state)

    i = 0
    while True:
        neighbours = create_neighbours(state)

        neighbours_cost = dict(zip(map(cost_function, neighbours), neighbours))
        better_cost = list(filter(lambda x: x > cost_state, neighbours_cost.keys()))

        if not better_cost:
            break
        state = neighbours_cost[better_cost[0]]
        cost_state = cost_function(state)
        print_step(neighbours_cost, i, state, cost_state)
        i += 1

    print("Etat " + str(state))
    print("Cost : " + str(cost_state))


best_improvment(initial_state)
first_improvment(initial_state)
random_improvment(initial_state)





