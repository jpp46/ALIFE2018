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
raw_scatter = True
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
    v2 = np.array([-1, -1])
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
        points.append([x, y, gens[i]])
        thetas.append(angle_to_xy([x, y]))
        lengths.append(np.sqrt(x**2 + y**2))
        if x <= 0 and y <= 0:
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
algor = ["Legged", "Wheeled", "Whegged"]
funct = ["Sum", "Product", "Min"]
seeds = range(30)

cmap = mpl.cm.winter
norm = mpl.colors.Normalize(vmin=0, vmax=3000)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

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
                _, d1 = compute_fitness(genome, alg, envs[0])
                _, d2 = compute_fitness(genome, alg, envs[2])
                c.write(str(max(d1, d2))+'\n')
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
if raw_scatter:
    fig, ax = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=True)
    for (fun, row, z) in zip(funs, ax, range(3)):
        for (alg, col, k) in zip(algorithms, row, range(3)):
            points = []

            for seed in tqdm(seeds):
                f = open(alg+"/"+"Results/"+fun+"_"+str(seed)+"_results.p", "rb")
                matrix = np.array(pickle.load(f))**(-(1/2))
                f.close()

                for i in range(len(matrix)-1):
                    for j in range(100):
                        fit = fitness(matrix[i, j, 0], matrix[i, j, 1], fun)
                        a = (matrix[i+1, j, 0] - matrix[i, j, 0])
                        b = (matrix[i+1, j, 1] - matrix[i, j, 1])
                        if not a == 0 and not b == 0:
                            points.append( (i, (a, b, i)) )

            points = sorted(points); l = len(points)
            x, y, gen = (np.array([points[i][1][1] for i in range(l)]),
                         np.array([points[i][1][0] for i in range(l)]),
                         np.array([points[i][1][2] for i in range(l)]))

            col.scatter(-x, -y, c=gen, cmap=cmap, norm=norm, facecolors='none', alpha=0.7)
            if z == 0:
                col.set_title(algor[k])
            if k == 0:
                col.set_ylabel(funct[z])

    #plt.xlim(-250, 250)
    #plt.ylim(-250, 250)
    #plt.xticks([-200, -100, 0, 100, 200])
    #plt.yticks([-200, -100, 0, 100, 200])
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0, wspace=0)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    plt.savefig("zGraphs/raw_results.jpg")
#################################################################################


#################################################################################
if best_scatter:
    fig, ax = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=True)
    theta_stats = []; length_stats = []; quadrant_stats = []
    for (fun, row, i) in zip(funs, ax, range(3)):
        for (alg, col, j) in zip(algorithms, row, range(3)):
            thetas = []; lengths = []; quadrants = []; points = []

            for seed in tqdm(seeds):
                f = open(alg+"/"+"Results/"+fun+"_"+str(seed)+"_results.p", "rb")
                matrix = np.array(pickle.load(f))**(-(1/2))
                f.close()

                theta, length, quadrant, point = ancestor_stats(matrix)
                thetas.append(np.mean(theta))
                lengths.append(np.mean(length))
                quadrants.append((np.sum(quadrant)/len(quadrant))*100)
                points += point

            theta_stats.append(thetas)
            #mu = np.mean(thetas); std = np.std(thetas)
            #print(fun+alg, "Mu:", str(mu),'', "Stddev: ", str(std))

            length_stats.append(lengths)
            #mu = np.mean(lengths); std = np.std(lengths)
            #print(fun+alg, "Mu:", str(mu),'', "Stddev: ", str(std))

            quadrant_stats.append(quadrants)
            #mu = np.mean(quadrants); std = np.std(quadrants)
            #print(fun+alg, "Mu:", str(mu),'', "Stddev: ", str(std))

            points = np.array(points)
            x = points[:, 0]; y = points[:, 1]; gen = points[:, 2]
            col.scatter(-x, -y, c=gen, cmap=cmap, norm=norm, facecolors='none', alpha=0.7)
            if i == 0:
                col.set_title(algor[j])
            if j == 0:
                col.set_ylabel(funct[i])


    compute_stats(theta_stats, "theta")
    print()
    compute_stats(length_stats, "length")
    print()
    compute_stats(quadrant_stats, "quadrants")

    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)
    #plt.xticks([-200, -100, 0, 100, 200])
    #plt.yticks([-200, -100, 0, 100, 200])
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0, wspace=0)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    plt.savefig("zGraphs/best_results.jpg")
#################################################################################
