import copy
import random
from itertools import combinations
import temp_sweep as sim
import scipy.stats as sp
import config as cf
import numpy as np
import os
import matplotlib.pyplot as plt

def scramble_Jij(emp_Jij, scale = 1):
    scrambled_matrix = copy.copy(emp_Jij)
    combo = list(combinations(set(range(84)), 2))
    random.shuffle(combo)

    num_shuffles = int(len(combo) * scale)
    num_shuffles = num_shuffles - (num_shuffles % 2)

    for i in np.arange(0, num_shuffles, step = 2):
        index_1 = combo[i]
        index_2 = combo[i + 1]

        scrambled_matrix[index_1] = scrambled_matrix[index_1[::-1]] = emp_Jij[index_2]
        scrambled_matrix[index_2] = scrambled_matrix[index_2[::-1]] = emp_Jij[index_1]

    return scrambled_matrix

def generate_samples(num_samples, steps, thermalization = None, spin_array = np.random.choice([-1, 1], 84).astype(np.int8), save = False):
    best_corr_array = []
    crit_corr_array = []
    for step in range(num_samples):
        print(step)
        rand_Jij = scramble_Jij(cf.avg_Jij, 1)
        simulation = sim.simulated_FC_vs_T_global(1, 3, 20, 0, Jij = rand_Jij)
        simulation.simulate(steps, thermalization = thermalization, spin_array = spin_array, text = False, partial = False)
        best_corr_array.append(np.round(simulation.best_corr, 2))
        crit_corr_array.append(np.round(simulation.crit_corr, 2))

    if save:
        directory = 'simulation data/bootstrap prob dist/'
        index = str(len(os.listdir(directory)) / 2)
        dir_name = 'bootstrap_' + index
        os.mkdir(directory + dir_name)
        with open(directory + dir_name + '/best_correlation.txt', 'w') as file:
            for i in best_corr_array:
                file.write(str(i) + ' ')

        with open(directory + dir_name + 'critical_correlation.txt', 'w') as file:
            for i in crit_corr_array:
                file.write(str(i) + ' ')

        with open(directory + dir_name + 'initial_spins.txt', 'w') as file:
            for i in spin_array:
                file.write(str(i) + ' ')

    return best_corr_array, crit_corr_array

def p_value_bootstrap(sample, iterations, emp_corr):
    num_samples = np.size(sample)
    p_values = [np.size(sample[sample >= emp_corr]) / np.size(sample)]

    for i in range(iterations):
        print(i)
        resample = sample[np.random.randint(0, num_samples, num_samples)]
        p_values.append(np.size(resample[resample >= emp_corr]) / np.size(resample))

    return p_values

file = open('simulation data/bootstrap prob dist/bootstrap_0.0/bootstrap_0.0best_correlation.txt', 'r')
samples = file.read().split(' ')
samples = np.array(list(map(np.float32, samples[:-1])))

file = open('simulation data/bootstrap prob dist/bootstrap_0.0/bootstrap_0.0initial_spins.txt', 'r')
spins = file.read().split(' ')
spins = np.array(list(map(np.int8, spins[:-1])))

emp_sim = sim.simulated_FC_vs_T_global(1, 3, 20, 0, Jij = cf.avg_Jij)
emp_sim.simulate(1000, spin_array = spins, text = False, partial = False)
emp_corr = emp_sim.best_corr
p_values = p_value_bootstrap(samples, 10000, emp_corr)
plt.hist(p_values, 20, (0, 0.02))
plt.xlabel('p value', fontsize = 12)
plt.ylabel('counts', fontsize = 12)
plt.show()