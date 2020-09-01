import pyrosim
import random
import math
import pickle
import sys

import numpy as np

from legrobot import Legged
from whegrobot import Whegged
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

def compute_fitness(genome, env, func):
    sim = new_sim(pb=True, pp=False)
    robot = func(sim, genome)
    robot.evaluate(sim, env)
    fitness, distance, x, y = robot.eval_fitness(sim, env)
    return fitness, distance, x, y

envs = init_evnironments(4)
A = envs[0]
B = envs[2]

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

f = open('WheelsAndJoints/Results/'+fun+'_'+str(seed)+'_genomes.p', 'rb')
matrix = pickle.load(f)
f.close()
whegged = matrix[-1][0]

_, _, x1, y1 = compute_fitness(whegged, A, Whegged)
_, _, x2, y2 = compute_fitness(whegged, B, Whegged)

x1 = [x1[i] for i in range(1, len(x1), 15)]
y1 = [y1[i] for i in range(1, len(y1), 15)]
x2 = [x2[i] for i in range(1, len(x2), 15)]
y2 = [y2[i] for i in range(1, len(y2), 15)]

plt.scatter(x1, y1, color='b', marker='s', facecolors='none', label='Whegged in Env A')
plt.scatter(x2, y2, color='b', marker='o', facecolors='none', label='Whegged in Env B')

f = open('Baseline/Results/'+fun+'_'+str(seed)+'_genomes.p', 'rb')
matrix = pickle.load(f)
f.close()
legged = matrix[-1][0]

_, _, x1, y1 = compute_fitness(legged, A, Legged)
_, _, x2, y2 = compute_fitness(legged, B, Legged)

x1 = [x1[i] for i in range(1, len(x1), 15)]
y1 = [y1[i] for i in range(1, len(y1), 15)]
x2 = [x2[i] for i in range(1, len(x2), 15)]
y2 = [y2[i] for i in range(1, len(y2), 15)]

plt.scatter(x1, y1, color='r', marker='s', facecolors='none', label='Legged in Env A')
plt.scatter(x2, y2, color='r', marker='o', facecolors='none', label='Legged in Env B')

plt.xlim(-10, 10)
plt.ylim(-3.5, 3.5)
plt.xlabel(r'$\Delta x$ position')
plt.ylabel(r'$\Delta y$ position')
plt.legend()

plt.show()
