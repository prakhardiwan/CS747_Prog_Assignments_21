from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter #https://stackoverflow.com/a/29188910
import bandit

instances1 = ["../instances/instances-task1/i-3.txt"]
algorithms1 = ["epsilon-greedy-t1","ucb-t1","kl-ucb-t1","thompson-sampling-t1"]
#sys.stdout = open('./output_dir/task1.txt', 'w')
horizons1 = [100, 400, 1600, 6400, 25600, 102400]

th = 0
ep = 0.02
c= 2

seeds = np.zeros((50,),dtype=int)
for i in range(50):
    seeds[i] = i
regrets = {}
#lin_arr = np.arange(hz + 1)
i = 1
for instance in instances1:
    regrets['epsilon-greedy-t1'] = np.zeros(6)
    regrets['ucb-t1'] = np.zeros(6)
    regrets['kl-ucb-t1'] = np.zeros(6)
    regrets['thompson-sampling-t1'] = np.zeros(6)
    ins = instance
    for algorithm in algorithms1:
        al = algorithm
        horizon_index = 0
        for hz in horizons1:
            reg_val = 0
            for seed in seeds:
                rs = seed
                reg_val+= bandit.Bandit(al, rs, ep, c, th, hz,ins)
                #res = b.run()
                # += (lin_arr * res.max_rew - res.cum_rew_hist)
            regrets[algorithm][horizon_index] = reg_val/50.0
            horizon_index+=1
            print("Hello horizon {}",hz)    
    #print(len(seeds))
    x = np.log10(horizons1)
    fig, ax = plt.subplots()
    plt.xscale('log')
    plt.xlabel('Horizon')
    plt.ylabel('Regret')
    plt.title(f'Plot for Task 1: Instance i-3')
    ax.plot(x, regrets['epsilon-greedy-t1'], '-r', label='epsilon-greedy-t1')
    ax.plot(x, regrets['kl-ucb-t1'], '-g', label='kl-ucb-t1')
    ax.plot(x, regrets['ucb-t1'], '-b', label='ucb-t1')
    ax.plot(x, regrets['thompson-sampling-t1'], '-k', label='thompson-sampling-t1')
    ax.legend(loc='upper left', frameon=False)
    leg = ax.legend()
    fig.savefig(f'./pics_dir/task1_i3_eps1e-4.png')

    i += 1

#sys.stdout.close()