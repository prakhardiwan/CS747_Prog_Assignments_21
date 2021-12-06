from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter #https://stackoverflow.com/a/29188910
import bandit

instances1 = ["../instances/instances-task3/i-1.txt", "../instances/instances-task3/i-2.txt"]
algorithms1 = ["alg-t3"]
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
    ins = instance
    for algorithm in algorithms1:
        al = algorithm
        horizon_index = 0
        for hz in horizons1:
            reg_val = 0
            for seed in seeds:
                rs = seed
                #reg_val+= bandit.Bandit(al, rs, ep, c, th, hz,ins)
                print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(ins, al, rs, ep, c, th, hz,0,0))
                #res = b.run()
                # += (lin_arr * res.max_rew - res.cum_rew_hist)
            #regrets[algorithm][horizon_index] = reg_val/50.0
            horizon_index+=1
    #print(len(seeds))

    i += 1

#sys.stdout.close()