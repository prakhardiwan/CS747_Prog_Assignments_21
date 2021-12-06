import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter #https://stackoverflow.com/a/29188910
import bandit
import os 
instances = ["../instances/instances-task2/i-1.txt", "../instances/instances-task2/i-2.txt", "../instances/instances-task2/i-3.txt", "../instances/instances-task2/i-4.txt", "../instances/instances-task2/i-5.txt"]
algorithm = "ucb-t2"
hz = 10000
randomSeed = np.zeros((50,),dtype=int)
for i in range(50):
    randomSeed[i] = i
epsilon  = 0.02
c_scale = np.zeros(15)
for i in range(15):
    c_scale[i] = 0.02*(i+1)
threshold = 0

for IN in instances:
    for sc in c_scale:
        for rs in randomSeed:
            os.system("python bandit.py --instance {} --algorithm {} --randomSeed {} --epsilon {} --scale {} --threshold {} --horizon {}\n".format(IN, algorithm, rs,epsilon, sc, threshold,hz))
