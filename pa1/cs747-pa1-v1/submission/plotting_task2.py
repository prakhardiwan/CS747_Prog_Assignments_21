from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter #https://stackoverflow.com/a/29188910
import bandit_saved

instances1 = ["../instances/instances-task2/i-1.txt", "../instances/instances-task2/i-2.txt", "../instances/instances-task2/i-3.txt", "../instances/instances-task2/i-4.txt", "../instances/instances-task2/i-5.txt"]
algorithm = "ucb-t2"
sys.stdout = open('./output_dir/task2.txt', 'w')
horizons1 = 10000

th = 0
ep = 0.02
c_scale = np.zeros(15)
for i in range(15):
    c_scale[i] = 0.02*(i+1)
seeds = np.zeros((50,),dtype=int)
for i in range(50):
    seeds[i] = i
regrets = {}
#lin_arr = np.arange(hz + 1)
regrets['../instances/instances-task2/i-1.txt'] = np.zeros(15)
regrets['../instances/instances-task2/i-2.txt'] = np.zeros(15)
regrets['../instances/instances-task2/i-3.txt'] = np.zeros(15)
regrets['../instances/instances-task2/i-4.txt'] = np.zeros(15)
regrets['../instances/instances-task2/i-5.txt'] = np.zeros(15)
for instance in instances1:
    ins = instance
    scale_index = 0
    for c in c_scale:
        reg_val = 0
        for seed in seeds:
            rs = seed
            reg_val+= bandit_saved.Bandit(algorithm, rs, ep, c, th, horizons1,ins)
            #res = b.run()
            # += (lin_arr * res.max_rew - res.cum_rew_hist)
        regrets[instance][scale_index] = reg_val/50.0
        scale_index+=1    
    #print(len(seeds))
x = c_scale
fig, ax = plt.subplots()
plt.xlabel('Scale')
plt.ylabel('Regret')
plt.title(f'Plot for Task 2')
ax.plot(x, regrets['../instances/instances-task2/i-1.txt'], '-g', label='Instance 1')
ax.plot(x, regrets['../instances/instances-task2/i-2.txt'], '-b', label='Instance 2')
ax.plot(x, regrets['../instances/instances-task2/i-3.txt'], '-k', label='Instance 3')
ax.plot(x, regrets['../instances/instances-task2/i-4.txt'], '-r', label='Instance 4')
ax.plot(x, regrets['../instances/instances-task2/i-5.txt'], '-m', label='Instance 5')
ax.legend(loc='upper left', frameon=False)
leg = ax.legend()
fig.savefig(f'./pics_dir/task2_final.png')
print(np.argmin(regrets['../instances/instances-task2/i-1.txt']))
print(np.argmin(regrets['../instances/instances-task2/i-2.txt']))
print(np.argmin(regrets['../instances/instances-task2/i-3.txt']))
print(np.argmin(regrets['../instances/instances-task2/i-4.txt']))
print(np.argmin(regrets['../instances/instances-task2/i-5.txt']))

sys.stdout.close()