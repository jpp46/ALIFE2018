import pyrosim
import random
import math
import pickle
import sys

import numpy as np

from robot import Robot
from environment import Environment
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

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
    sim = new_sim(pb=True, pp=False)
    robot = Robot(sim, genome)
    robot.evaluate(sim, env)
    fitness, distance, x, y = robot.eval_fitness(sim, env)
    return fitness, distance, x, y

envs = init_evnironments(4)
A = envs[0]
B = envs[2]

f = open('Results/'+fun+'_'+str(seed)+'_genomes.p', 'rb')
matrix = pickle.load(f)
f.close()

print(len(matrix))
genome = matrix[-1][0]
dist = matrix[-1][1]
fit = matrix[-1][2]


_, _, x1, y1 = compute_fitness(genome, A)
_, _, x2, y2 = compute_fitness(genome, B)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.show()
