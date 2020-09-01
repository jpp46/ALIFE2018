import pyrosim
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from legrobot import Legged
from wheelrobot import Wheeled
from whegrobot import Whegged
from environment import Environment

from tqdm import tqdm
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

def init_evnironments(num):
    envs = []
    for i in range(num):
        envs.append(Environment(i))
    return np.array(envs)

def new_sim(pp=False, pb=True, t=1000):
    return pyrosim.Simulator(play_paused=pp, play_blind=pb, eval_time=t, xyz=[0, 0, 12], hpr=[90, -90, 0])

def compute_fitness(genome, ty, env):
    sim = new_sim()
    robot = None

    if ty == "Baseline":
        robot = Legged(sim, genome)
    if ty == "AllWheels":
        robot = Wheeled(sim, genome)
    if ty == "WheelsAndJoints":
        robot = Whegged(sim, genome)

    robot.evaluate(sim, env)
    fitness, distance = robot.eval_fitness(sim, env)
    return fitness, distance

csv = True
best_scatter = True

envs = init_evnironments(4)

def fitness(a, b, fun):
    if fun == "*":
        return a * b
    if fun == "+":
        return a + b
    if fun == "min":
        return min(a, b)

def unit_vector(vector):
    if np.sum(vector) == 0:
        return vector
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    theta = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(theta)

def angle_to_xy(v1):
    v1 = np.array(v1)
    v2 = np.array([-1, -1, -1, -1])
    theta = angle_between(v1, v2)
    if theta > 180.0:
        theta = np.abs(theta - 360)
    return theta

def ancestor_stats(matrix):
    lineage = []; gens = []
    arr = np.max(matrix, axis=2)
    idx = np.unravel_index(np.argmin(arr), np.shape(arr))

    lineage.append(matrix[idx[0], idx[1]]); gens.append(idx[0])
    child = arr[idx[0], idx[1]]
    while not idx[0] == 0:
        parent = arr[idx[0], idx[1]]
        if not parent == child:
            lineage.append(matrix[idx[0], idx[1]])
            gens.append(idx[0])
            child = parent
        idx = (idx[0]-1, idx[1])
    if len(lineage) == 1:
        lineage.append(lineage[0])
        gens.append(gens[0])

    thetas = []; lengths = []; quadrants = []; points = []
    for i in range(len(lineage)-1):
        y = lineage[i][0] - lineage[i+1][0]
        x = lineage[i][1] - lineage[i+1][1]
        z = lineage[i][2] - lineage[i+1][2]
        k = lineage[i][3] - lineage[i+1][3]
        points.append([x, y, z, k, gens[i]])
        thetas.append(angle_to_xy([x, y, z, k]))
        lengths.append(np.sqrt(x**2 + y**2 + z**2 + k**2))
        if x <= 0 and y <= 0 and z <= 0 and k <= 0: # 1 is in top right quadrant 0 is elsewhere
            quadrants.append(1)
        else:
            quadrants.append(0)

    return thetas, lengths, quadrants, points

def compute_stats(data, name):
    data_table = np.reshape(data, (3, 3, 30))
    rp_table = np.zeros((3, 3))
    fp_table = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            if j == 2:
                rp_table[i, j] = mannwhitneyu(data_table[i, j], data_table[i, 0]).pvalue
            else:
                rp_table[i, j] = mannwhitneyu(data_table[i, j], data_table[i, j+1]).pvalue

    for i in range(3):
        for j in range(3):
            if i == 2:
                fp_table[i, j] = mannwhitneyu(data_table[i, j], data_table[0, j]).pvalue
            else:
                fp_table[i, j] = mannwhitneyu(data_table[i, j], data_table[i+1, j]).pvalue

    cols = ["Legged", "Wheeled", "Whegged"]
    p_cols = ["Legged-Wheeled", "Wheeled-Whegged", "Wheeged-Legged"]
    ind = ["Sum", "Product", "Min"]
    p_ind = ["Sum-Product", "Product-Min", "Min-Sum"]
    mean_table = pd.DataFrame(np.mean(data_table, axis=2), columns=cols, index=ind)
    std_table = pd.DataFrame(np.std(data_table, axis=2), columns=cols, index=ind)
    med_table = pd.DataFrame(np.median(data_table, axis=2), columns=cols, index=ind)
    rp_table = pd.DataFrame(rp_table*18, columns=p_cols, index=ind)
    fp_table = pd.DataFrame(fp_table*18, columns=cols, index=p_ind)

    f = open("zGraphs/%stable.txt" % name, "w")
    f.write("____MEAN____\n\n"+
                     mean_table.to_latex(bold_rows=True)+
                     "\n\n")
    f.write("____STDDEV____\n\n"+
                     std_table.to_latex(bold_rows=True)+
                     "\n\n")
    f.write("____MEDIAN____\n\n"+
                     med_table.to_latex(bold_rows=True)+
                     "\n\n")
    f.write("____ROBOT-P____\n\n"+
                     rp_table.to_latex(bold_rows=True)+
                     "\n\n")
    f.write("____FUNCT-P____\n\n"+
                     fp_table.to_latex(bold_rows=True)+
                     "\n\n")
    f.close()


algorithms = ["Baseline", "AllWheels", "WheelsAndJoints"]
funs = ["+", "*", "min"]
seeds = range(30)

cmap = mpl.cm.Reds
norm = mpl.colors.Normalize(vmin=0, vmax=3000)

#################################################################################
if csv:
    for fun in funs:
        for alg in algorithms:
            c = open("xResults/"+alg+"_"+fun+".csv", 'w')
            for seed in tqdm(seeds):
                g = open(alg+'/'+'Results/'+fun+'_'+str(seed)+'_genomes.p', 'rb')
                matrix = pickle.load(g)
                g.close()
                genome = matrix[-1][0]
                d = []
                for env in envs:
                    _, d0 = compute_fitness(genome, alg, env)
                    d.append(d0)
                c.write(str(max(d))+'\n')
            c.close()

    stats = []
    for fun in funs:
        for alg in algorithms:
            data = np.genfromtxt('xResults/'+alg+'_'+fun+'.csv', delimiter='\n')
            stats.append(data)
            #mu = np.mean(data); std = np.std(data);
            #print(fun+alg, "Mu:", str(mu),'', "Stddev: ", str(std))

    compute_stats(stats, "distance")
#################################################################################


#################################################################################
if best_scatter:
    theta_stats = []; length_stats = []; quadrant_stats = [];
    for fun in funs:
        for alg in algorithms:
            thetas = []; lengths = []; quadrants = []; points = []

            for seed in tqdm(seeds):
                f = open(alg+"/"+"Results/"+fun+"_"+str(seed)+"_results.p", "rb")
                matrix = np.array(pickle.load(f))**(-(1/2))
                f.close()

                theta, length, quadrant, _ = ancestor_stats(matrix)
                thetas.append(np.mean(theta))
                lengths.append(np.mean(length))
                quadrants.append((np.sum(quadrant)/len(quadrant))*100)

            theta_stats.append(thetas)
            #mu = np.mean(thetas); std = np.std(thetas)
            #print(fun+alg, "Mu:", str(mu),'', "Stddev: ", str(std))

            length_stats.append(lengths)
            #mu = np.mean(lengths); std = np.std(lengths)
            #print(fun+alg, "Mu:", str(mu),'', "Stddev: ", str(std))

            quadrant_stats.append(quadrants)
            #mu = np.mean(quadrants); std = np.std(quadrants)
            #print(fun+alg, "Mu:", str(mu),'', "Stddev: ", str(std))

    compute_stats(theta_stats, "theta")
    print()
    compute_stats(length_stats, "length")
    print()
    compute_stats(quadrant_stats, "quadrants")
#################################################################################
