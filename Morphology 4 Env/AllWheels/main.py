import pyrosim
import random
import math
import pickle
import sys

import numpy as np

from robot import Robot
from environment import Environment
from tqdm import tqdm

envs_num = 4
pop_size = 100
generations = 3000
mutation_rate = 0.05
batch_size = 10

def save(population, fits, dist, tag, seed):
    idx = np.argmin(dist)

    f = open('Results/'+tag+'_'+str(seed)+'_genomes.p', 'rb')
    matrix = pickle.load(f)
    f.close()

    results = [population[idx], dist[idx], fits[idx]]
    matrix.append(results)

    f = open('Results/'+tag+'_'+str(seed)+'_genomes.p', 'wb')
    pickle.dump(matrix, f)
    f.close()

def save_results(A, B, C, D, tag, seed):
    f = open('Results/'+tag+'_'+str(seed)+'_results.p', 'rb')
    matrix = pickle.load(f)
    f.close()

    results = np.zeros((pop_size, 4))
    results[:, 0] = A[0]; results[:, 1] = B[0]; results[:, 2] = C[0]; results[:, 3] = D[0]
    matrix.append(results)

    f = open('Results/'+tag+'_'+str(seed)+'_results.p', 'wb')
    pickle.dump(matrix, f)
    f.close()

def init_evnironments(num):
    envs = []
    for i in range(num):
        envs.append(Environment(i))
    return np.array(envs)

def new_sim(pp=False, pb=True, t=1000):
    return pyrosim.Simulator(play_paused=pp, play_blind=pb, eval_time=t)

def compute_fitness(population, env):
    sims = []; fits = []; dist = [];
    for i in range((pop_size // batch_size)):
        shift = i*10
        for j in range(batch_size):
            sims.append(new_sim())
            robot = Robot(sims[j+shift], population[j+shift])
            robot.evaluate(sims[j+shift], env)

        for j in range(batch_size):
            fitness, distance = robot.eval_fitness(sims[j+shift], env)
            fits.append(fitness); dist.append(distance)

    return np.array(fits), np.array(dist)

def mutants(population):
    mu = population; sigma = mutation_rate
    children = sigma * np.random.randn(pop_size, 5, 8) + mu
    return children

def prioritized(population, fits):
    cs = np.cumsum(fits)
    rnum = np.random.random(pop_size)*cs[-1]
    idx = np.searchsorted(cs, rnum, side='left')
    return [population[i] for i in idx]

def offspring(most_fit_parent, parent):
    child = np.zeros((5, 8))
    for i in range(5):
        for j in range(8):
            if random.randint(0, 1) == 0:
                child[i, j] = most_fit_parent[i, j]
            else:
                child[i, j] = parent[i, j]
    return child

def reproduction(population, fits):
    most_fit = prioritized(population, fits)
    children = np.array([offspring(random.choice(most_fit), population[i]) for i in range(pop_size)])
    return children

def survival_of_fitest(parents, parent_fits, parent_dist, A, B, C, D, children, child_fits, child_dist, child_A, child_B, child_C, child_D):
    for i in range(pop_size):
        if child_fits[i] > parent_fits[i]:
            parents[i] = children[i]
            parent_fits[i] = child_fits[i]
            parent_dist[i] = child_dist[i]
            A[0][i] = child_A[0][i]
            A[1][i] = child_A[1][i]
            B[0][i] = child_B[0][i]
            B[1][i] = child_B[1][i]
            C[0][i] = child_C[0][i]
            C[1][i] = child_C[1][i]
            D[0][i] = child_D[0][i]
            D[1][i] = child_D[1][i]
    return parents, parent_fits, parent_dist, A, B, C, D

def fitness(tag, comp1, comp2, comp3, comp4):
    f1, d1 = comp1
    f2, d2 = comp2
    f3, d3 = comp3
    f4, d4 = comp4
    if tag == "*":
        return (f1 * f2 * f3 * f4, d1 + d2 + d3 + d4)
    elif tag == "+":
        return (f1 + f2 + f3 + f4, d1 + d2 + d3 + d4)
    elif tag == "min":
        f = np.minimum(f1, f2)
        f = np.minimum(f, f3)
        f = np.minimum(f, f4)
        return (f, d1 + d2 + d3 + d4)


tag = sys.argv[1]
seed = int(sys.argv[2])

random.seed(seed)
np.random.seed(seed)

matrix = []
f = open('Results/'+tag+'_'+str(seed)+'_genomes.p', 'wb')
pickle.dump(matrix, f)
f.close()
f = open('Results/'+tag+'_'+str(seed)+'_results.p', 'wb')
pickle.dump(matrix, f)
f.close()

envs = init_evnironments(envs_num)
parents = np.random.random((pop_size, 5, 8)) * 2.0 - 1.0
A = compute_fitness(parents, envs[0])
B = compute_fitness(parents, envs[1])
C = compute_fitness(parents, envs[2])
D = compute_fitness(parents, envs[3])
parent_fits, parent_dist = fitness(tag, A, B, C, D)
save_results(A, B, C, D, tag, seed)

for g in tqdm(range(generations)):
    children = mutants(parents)
    child_A = compute_fitness(children, envs[0])
    child_B = compute_fitness(children, envs[1])
    child_C = compute_fitness(children, envs[2])
    child_D = compute_fitness(children, envs[3])
    child_fits, child_dist = fitness(tag, child_A, child_B, child_C, child_D)
    parents, parent_fits, parent_dist, A, B, C, D = survival_of_fitest(parents, parent_fits, parent_dist, A, B, C, D, children, child_fits, child_dist, child_A, child_B, child_C, child_D)

    save(parents, parent_fits, parent_dist, tag, seed)
    save_results(A, B, C, D, tag, seed)
