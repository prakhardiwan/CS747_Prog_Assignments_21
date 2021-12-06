import argparse
import random
import numpy as np
import math

# Function Definitions
def pull_arm_gen_reward(arm_to_pull,prob_val): # (x, p[x])
    # takes in the arm to pull and that arm's prob value
    if(np.random.uniform(0,1.0)<prob_val):
        return 1
    else:
        return 0
    # this function randomely generates the reward 0 or 1 as per the arm's 

#Choosing the arm functions for different algos
def choose_arm_eps(total_arms, epsilon, emperical_mean):
    if(np.random.uniform(0,1.0)<epsilon):
        return np.random.randint(0,total_arms)
    else:
        return np.argmax(emperical_mean)

def compute_UCB(T,num_pulls_for_arm,emperical_mean,total_arms): 
    delta = 0.0001
    UCB_calc = np.zeros(total_arms)
    sqrt_vals = np.sqrt((2*(math.log(T)))/(num_pulls_for_arm+delta))
    for i in range(len(UCB_calc)):
        UCB_calc[i] = emperical_mean[i] + sqrt_vals[i]
    return UCB_calc

def choose_arm_UCB(UCB):
    return np.argmax(UCB)

def KL(p,q):
    if p==0:
        return (1-p)*np.log((1-p)/(1-q))
    elif p==1:
        return p*np.log(p/q)
    else:
        return (p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q)))

def find_qmax(upper_limit, num_pulls, emp_mean):
    right_val = 0.9999
    left_val = emp_mean
    q = right_val
    while(left_val<=right_val):
        q = (left_val+right_val)/2
        if((num_pulls*KL(emp_mean,q))<=upper_limit):
            left_val = q+1e-10
        else:
            right_val = q-1e-10
    return q

def compute_KL_UCB(T,num_pulls_for_arm,emperical_mean,numArms):
    delta = 0.0001
    Vals_KL_UCB = np.zeros(numArms)
    upper_limit = np.log(T+delta)+3*np.log(np.log(T+delta)) # C taken as 3
    for i in range(numArms):
        Vals_KL_UCB[i] = find_qmax(upper_limit,num_pulls_for_arm[i],emperical_mean[i])
    return Vals_KL_UCB

def choose_arm_KL_UCB(KL_UCB):
    return np.argmax(KL_UCB)

def  Thompson_Sampling(num_pulls_arm,numArms,reward_for_arm):
    t_samples = np.zeros(numArms)
    for i in range(numArms):
        num_failures = num_pulls_arm[i]-reward_for_arm[i]
        t_samples[i] = np.random.beta(reward_for_arm[i]+1,num_failures+1)
    return t_samples

def choose_arm_TS(Thompson_Samples):
    return np.argmax(Thompson_Samples)

def compute_UCB_with_scale(T,num_pulls_for_arm,emperical_mean,total_arms,scale):
    delta = 0.0001
    UCB_calc = np.zeros(total_arms)
    sqrt_vals = np.sqrt((scale*(math.log(T)))/(num_pulls_for_arm+delta))
    for i in range(len(UCB_calc)):
        UCB_calc[i] = emperical_mean[i] + sqrt_vals[i]
    return UCB_calc

# Extracting Inputs 
#parser = argparse.ArgumentParser(description='Taking inputs')
#parser.add_argument("--instance", help="Instance Path", type=str, default="../instances/instances-task1/i-1.txt")
#parser.add_argument("--algorithm", help="Algorithm", type=str, default="epsilon-greedy-t1")
#parser.add_argument("--randomSeed", help="Random Seed", type=int, default = 1)
#parser.add_argument("--epsilon", help="Epsilon", type=float, default = 0.02)        # For everything except epsilon-greedy, pass 0.02.
#parser.add_argument("--scale", help="Scale", type=float, default = 2)            # The parameter is only relevant for Task 2; for other tasks pass --scale 2
#parser.add_argument("--threshold", help="Threshold", type=float, default = 0)    # The parameter is only relevant for Task 4; for other tasks pass --threshold 0.
#parser.add_argument("--horizon", help="Horizon", type=int, default = 30)
def Bandit(al, rs, ep, c, th, hz,ins):
    #inp = parser.parse_args()
    np.random.seed(rs)
    scale = c
    with open(ins) as probs:
        p = probs.readlines()      # p is the probability array for arms
    for i in range(len(p)):
        p[i] = float(p[i])
    numArms = len(p)            # number of arms
    p = np.array(p)

    emperical_mean = np.zeros(numArms)
    num_pulls_for_arm = np.zeros(numArms)
    reward_for_arm = np.zeros(numArms)
    total_reward = 0

    total_pulls = hz
    eps = ep
    # Going over for pulling 

    if al == "epsilon-greedy-t1":
        #print("Epsilon Greedy - Task 1")
        for iter in range(0,total_pulls):
            arm = (choose_arm_eps(numArms,eps,emperical_mean))
            num_pulls_for_arm[arm] += 1
            reward_for_arm[arm] += pull_arm_gen_reward(arm,p[arm])
            total_reward += pull_arm_gen_reward(arm,p[arm])
            emperical_mean[arm] = reward_for_arm[arm]/num_pulls_for_arm[arm]
        #print(total_reward)
        HIGHS = 0
    elif al == "ucb-t1":
        #print("UCB - Task 1")
        for iter in range(0,total_pulls):
            UCB = compute_UCB(iter+1,num_pulls_for_arm,emperical_mean,numArms)
            arm = choose_arm_UCB(UCB)
            num_pulls_for_arm[arm] += 1
            reward_for_arm[arm] += pull_arm_gen_reward(arm,p[arm])
            total_reward += pull_arm_gen_reward(arm,p[arm])
            emperical_mean[arm] = reward_for_arm[arm]/num_pulls_for_arm[arm]
        #print(total_reward)
        HIGHS = 0
    elif al == "kl-ucb-t1":
        #print("KL-UCB - Task 1") 
        for iter in range(0,total_pulls):
            KL_UCB = compute_KL_UCB(iter+1,num_pulls_for_arm,emperical_mean,numArms)
            arm = choose_arm_KL_UCB(KL_UCB)
            num_pulls_for_arm[arm] += 1
            reward_for_arm[arm] += pull_arm_gen_reward(arm,p[arm])
            total_reward += pull_arm_gen_reward(arm,p[arm])
            emperical_mean[arm] = reward_for_arm[arm]/num_pulls_for_arm[arm]
        #print(total_reward)
        HIGHS = 0
    elif al == "thompson-sampling-t1":
        #print("Thompson Sampling - Task 1") 
        for iter in range(0,total_pulls):
            Thompson_Samples = Thompson_Sampling(num_pulls_for_arm,numArms,reward_for_arm)
            arm = choose_arm_TS(Thompson_Samples)
            num_pulls_for_arm[arm] += 1
            reward_for_arm[arm] += pull_arm_gen_reward(arm,p[arm])
            total_reward += pull_arm_gen_reward(arm,p[arm])
            emperical_mean[arm] = reward_for_arm[arm]/num_pulls_for_arm[arm]
        #print(total_reward)
        HIGHS = 0
    elif al == "ucb-t2":
        #print("UCB - Task 2")
        for iter in range(0,total_pulls):
            UCB = compute_UCB_with_scale(iter+1,num_pulls_for_arm,emperical_mean,numArms,scale)
            arm = choose_arm_UCB(UCB)
            num_pulls_for_arm[arm] += 1
            reward_for_arm[arm] += pull_arm_gen_reward(arm,p[arm])
            total_reward += pull_arm_gen_reward(arm,p[arm])
            emperical_mean[arm] = reward_for_arm[arm]/num_pulls_for_arm[arm]
        #print(total_reward)
        HIGHS = 0
    else:
        #print("NOT DONE")
        HIGHS = 0

    #print(p)
    #total_pulls = float(total_pulls)
    REG = total_pulls*(np.amax(p))-total_reward
    # Thing to be printed finally
    print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(ins, al, rs,eps, c, th,hz,REG,HIGHS))
    #print(REG)
    #print(numArms)
    #print(inp.instance)
    return REG


