import pyrosim
import random
import math
import pickle
import sys

import numpy as np

from robot import Robot
from environment import Environment
from tqdm import tqdm

fun = sys.argv[1]
seed = int(sys.argv[2])

def init_evnironments(num):
    envs = []
    for i in range(num):
        envs.append(Environment(i))
    return np.array(envs)

def new_sim(pp=False, pb=True, t=1000):
    return pyrosim.Simulator(play_paused=pp, play_blind=pb, eval_time=t, xyz=[0, 0, 12], hpr=[90, -90, 0])

def compute_fitness(genome, env):
    sim = new_sim(pb=False)
    robot = Robot(sim, genome)
    robot.evaluate(sim, env)
    fitness, distance = robot.eval_fitness(sim, env)
    return fitness, distance

envs = init_evnironments(4)

f = open('Results/'+fun+'_'+str(seed)+'_genomes.p', 'rb')
matrix = pickle.load(f)
f.close()

print(len(matrix))
genome = matrix[-1][0]

for env in envs:
    print(compute_fitness(genome, env))
