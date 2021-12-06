import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter #https://stackoverflow.com/a/29188910
import bandit
import os 

instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]
algorithms = ["epsilon-greedy-t1","ucb-t1","kl-ucb-t1","thompson-sampling-t1"]
horizons1 = [100, 400, 1600, 6400, 25600, 102400]
randomSeed = np.zeros((50,),dtype=int)
for i in range(50):
    randomSeed[i] = i
epsilon  = 0.02
threshold = 0
scale = 2

for IN in instances:
    for al in algorithms:
        for hz in horizons1:
            for rs in randomSeed:
                os.system("python bandit.py --instance {} --algorithm {} --randomSeed {} --epsilon {} --scale {} --threshold {} --horizon {}".format(IN, al, rs,epsilon, scale, threshold,hz))
