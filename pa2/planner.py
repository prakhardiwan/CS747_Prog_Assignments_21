import argparse
import random
import numpy as np
import math
import pulp
from pulp import *
from numpy.linalg import inv
# Function Definitions
# epsilon is the precision set for the V_t+1 - V_t norm squared
def compute_val_function_VI(init_val_function, num_states, num_actions, gamma, Tran_prob, Reward, epsilon):
    next_valfn = np.zeros(num_states)
    prev_valfn = init_val_function
    #iter = 0
    while 1>0: #starting an infinite loop here, will terminate only when v_star is found
        #print(iter)
        Q = np.zeros((num_states,num_actions))
        #for s1 in range(num_states):
        #    for a in range(num_actions):
        #        Q[s1][a] = sum((Tran_prob[s1][a][s2]*(Reward[s1][a][s2] + gamma*prev_valfn[s2])) for s2 in range(num_states))
        Q = np.sum(Tran_prob*(Reward + gamma*prev_valfn),axis=2) #optimization
        for i in range(num_states):
            next_valfn[i] = max(Q[i])
        if(sum((next_valfn-prev_valfn)**2))<epsilon:
            return next_valfn
        prev_valfn = next_valfn.copy()
        #iter += 1
    #print("Q run was unsuccessfull")   
    return next_valfn

def compute_val_function_VI_EPI(init_val_function, num_states, num_actions, gamma, Tran_prob, Reward, epsilon, terminal_states, num_states_operable):
    next_valfn = np.zeros(num_states)
    prev_valfn = init_val_function
    #iter = 0
    while 1>0: #starting an infinite loop here, will terminate only when v_star is found
        #print(iter)
        Q = np.zeros((num_states,num_actions)) ##
        #for s1 in num_states_operable:
        #    for a in range(num_actions):
        #        Q[s1][a] = sum((Tran_prob[s1][a][s2]*(Reward[s1][a][s2] + gamma*prev_valfn[s2])) for s2 in range(num_states))
        Q = np.sum(Tran_prob*(Reward + gamma*prev_valfn),axis=2) #optimization
        for i in num_states_operable:
            next_valfn[i] = max(Q[i])
        for j in terminal_states:
            next_valfn[j] = 0
        if(sum((next_valfn-prev_valfn)**2))<epsilon:
            return next_valfn
        prev_valfn = next_valfn.copy()
        #iter += 1
    #print("Q run was unsuccessfull")   
    return next_valfn
def eval_policy_Cont(num_states, num_actions, gamma, Tran_prob, Reward,Policy):
    A = np.zeros((num_states,num_states))
    for i in range(num_states):
        for j in range(num_states):
            if i==j:
                A[i][j] = 1-gamma*Tran_prob[i][Policy[i]][j]
            else:
                A[i][j] = -1*gamma*Tran_prob[i][Policy[i]][j]
    B = np.zeros((num_states,1))
    for i in range(num_states):
        B[i][0] = sum(Tran_prob[i][Policy[i]][s2]*(Reward[i][Policy[i]][s2]) for s2 in range(num_states))
    soln = np.zeros(num_states)
    X = np.matmul(inv(A),B)
    for i in range(num_states):
        soln[i] = X[i][0]
    return soln
def eval_policy_Epi(num_states, num_actions, gamma, Tran_prob, Reward,Policy,terminal_states,num_states_operable):
    A = np.zeros((num_states,num_states))
    for i in range(num_states):
        for j in range(num_states):
            if i==j:
                A[i][j] = 1-gamma*Tran_prob[i][Policy[i]][j]
            else:
                A[i][j] = -1*gamma*Tran_prob[i][Policy[i]][j]
    for j in terminal_states:
        for i in range(num_states):
            if i==j :
                A[i][j] = 1
            else:
                A[i][j] = 0
    B = np.zeros((num_states,1))
    for i in range(num_states):
        B[i][0] = sum(Tran_prob[i][Policy[i]][s2]*(Reward[i][Policy[i]][s2]) for s2 in range(num_states))
    for j in terminal_states:
        B[j][0] = 0
    soln = np.zeros(num_states)
    X = np.matmul(inv(A),B)
    for i in range(num_states):
        soln[i] = X[i][0]
    return soln

def Improve_policy(num_states, num_actions, gamma, Tran_prob, Reward,eval_Q, prev_policy):
    eps2 = 1e-6
    policy_improved = prev_policy.copy()
    for s1 in range(num_states):
        for a in range(num_actions):
            q_val = sum((Tran_prob[s1][a][s2]*(Reward[s1][a][s2] + gamma*eval_Q[s2])) for s2 in range(num_states))
            if(q_val>(eval_Q[s1]+eps2)):
                policy_improved[s1] = a
    return policy_improved

def compute_val_fn_HPI(num_states, num_actions, gamma, Tran_prob, Reward):
    prev_policy = np.zeros(num_states, dtype=int)
    next_policy = np.zeros(num_states, dtype=int)
    val_func = np.zeros(num_states)
    while 1>0:
        eval_Q = eval_policy_Cont(num_states, num_actions, gamma, Tran_prob, Reward,prev_policy)
        next_policy = Improve_policy(num_states, num_actions, gamma, Tran_prob, Reward,eval_Q, prev_policy)
        if (next_policy==prev_policy).all():
            val_func = eval_policy_Cont(num_states, num_actions, gamma, Tran_prob, Reward,next_policy)
            return val_func
        prev_policy = next_policy.copy()
    return val_func

def compute_val_fn_HPI_EPI(num_states, num_actions, gamma, Tran_prob, Reward, terminal_states, num_states_operable):
    prev_policy = np.zeros(num_states, dtype=int) 
    next_policy = np.zeros(num_states, dtype=int)
    val_func = np.zeros(num_states)
    while 1>0:
        eval_Q = eval_policy_Epi(num_states, num_actions, gamma, Tran_prob, Reward,prev_policy,terminal_states,num_states_operable)
        next_policy = Improve_policy(num_states, num_actions, gamma, Tran_prob, Reward,eval_Q, prev_policy)
        if (next_policy==prev_policy).all():
            val_func = eval_policy_Epi(num_states, num_actions, gamma, Tran_prob, Reward,next_policy,terminal_states,num_states_operable)
            return val_func
        prev_policy = next_policy.copy()
    return val_func
############ Main Program #################################################################################

# Extracting Inputs 
parser = argparse.ArgumentParser(description='Taking inputs')
parser.add_argument("--mdp", help="Instance Path", type=str, default="data/mdp/continuing-mdp-2-2.txt")
parser.add_argument("--algorithm", help="Algorithm", type=str, default="vi")

inp = parser.parse_args()

with open(inp.mdp) as text_data:
    line_read = text_data.readlines()      # line_read is the lines read from text file

line_r = line_read[0].split()
num_states = int(line_r[1])
num_states_operable = []
for alpha in range(num_states):
    num_states_operable.append(alpha)
line_r = line_read[1].split()
num_actions = int(line_r[1])
Tran_prob = np.zeros((num_states,num_actions,num_states))
Reward = np.zeros((num_states,num_actions,num_states))
terminal_states = []
#print(num_states)
#print(num_actions)
for i in range(len(line_read)):
    line_r = line_read[i].split()
    if line_r[0] == "transition":
        state1 = int(line_r[1])
        state2 = int(line_r[3])
        action = int(line_r[2])
        Reward[state1][action][state2] = np.float64(line_r[4])
        Tran_prob[state1][action][state2] += np.float64(line_r[5]) # += for incorporating merged terminal states in task 2
    elif line_r[0] == "mdptype":
        MDP_Type = line_r[1]
    elif line_r[0] == "discount":
        gamma = float(line_r[1])
    elif line_r[0] == "end":
        n = len(line_r)
        #print(n)
        for k in range(1,n):
            terminal_states.append(int(line_r[k]))
#print(Reward)
#print(Tran_prob)
#print(terminal_states)
if MDP_Type == "episodic":
    num_states_operable = list(set(num_states_operable).difference(terminal_states))

init_val_function = np.zeros(num_states)
#print("Number of States is: {}".format(num_states))
#print("Number of Actions is: {}".format(num_actions))
#print("End States are: ")
#print(terminal_states)
#for i in range(num_states):
#    for j in range(num_actions):
#        for k in range(num_states):
#            print("Transition {} {} {}: Reward:{} Tran:{}".format(i,j,k,Reward[i][j][k],Tran_prob[i][j][k]))
# Calling the required function as per the algorithm
if inp.algorithm == "vi":
    #print("Reached")
    epsilon = 1e-18
    if MDP_Type == "continuing":
        val_function = compute_val_function_VI(init_val_function, num_states, num_actions, gamma, Tran_prob, Reward, epsilon)
    elif MDP_Type == "episodic":
        val_function = compute_val_function_VI_EPI(init_val_function, num_states, num_actions, gamma, Tran_prob, Reward, epsilon, terminal_states, num_states_operable)
    Q_val = np.zeros((num_states,num_actions))
    #for s1 in range(num_states):
    #    for a in range(num_actions):
    #        Q_val[s1][a] = sum((Tran_prob[s1][a][s2]*(Reward[s1][a][s2] + gamma*val_function[s2])) for s2 in range(num_states))
    Q_val = np.sum(Tran_prob*(Reward + gamma*val_function),axis=2) #optimization
    pi_star = np.argmax(Q_val,axis=1)
    for s1 in range(num_states):
        print ("{} {}".format(val_function[s1], pi_star[s1]))
    #print(val_function)
    #print("You called Value Iteration")
elif inp.algorithm == "hpi":
    val_fn_H = np.zeros(num_states)
    if MDP_Type == "continuing":
        val_fn_H = compute_val_fn_HPI(num_states, num_actions, gamma, Tran_prob, Reward)
    elif MDP_Type == "episodic":
        val_fn_H = compute_val_fn_HPI_EPI(num_states, num_actions, gamma, Tran_prob, Reward, terminal_states, num_states_operable)
    Q_val = np.zeros((num_states,num_actions))
    #for s1 in range(num_states):
    #    for a in range(num_actions):
    #        Q_val[s1][a] = sum((Tran_prob[s1][a][s2]*(Reward[s1][a][s2] + gamma*val_fn_H[s2])) for s2 in range(num_states))
    Q_val = np.sum(Tran_prob*(Reward + gamma*val_fn_H),axis=2) #optimization
    pi_star = np.argmax(Q_val,axis=1)
    for s1 in range(num_states):
        print ("{} {}".format(val_fn_H[s1], pi_star[s1]))

    #print("You called Howard's Policy Iteration")
elif inp.algorithm == "lp":
    prob = LpProblem("FindingOptimalV", LpMinimize)
    val_fn_LP = [0]*(num_states)
    for i in range(num_states):
        val_fn_LP[i] = LpVariable(str(i))
    # Objective Function : To be minimized
    prob += lpSum(val_fn_LP[i] for i in range(len(val_fn_LP))), "Objective Function"
    # Constraints:
    if MDP_Type == "episodic":
        for sT in terminal_states:
            prob += val_fn_LP[sT] == 0
    for s1 in num_states_operable:
        for a in range(num_actions):
            prob += val_fn_LP[s1] >= lpSum((Tran_prob[s1][a][s2]*(Reward[s1][a][s2]+gamma*val_fn_LP[s2])) for s2 in range(num_states))
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    V_star_LP = np.zeros(num_states) 
    for s1 in range(num_states):
	    V_star_LP[s1] = pulp.value(val_fn_LP[s1])
    Q_val = np.zeros((num_states,num_actions))
    #for s1 in range(num_states):
    #    for a in range(num_actions):
    #        Q_val[s1][a] = sum((Tran_prob[s1][a][s2]*(Reward[s1][a][s2] + gamma*pulp.value(val_fn_LP[s2]))) for s2 in range(num_states))
    Q_val = np.sum(Tran_prob*(Reward + gamma*V_star_LP),axis=2) #optimization
    pi_star = np.argmax(Q_val,axis=1)
   
    for s1 in range(num_states):
        print ("{} {}".format(V_star_LP[s1], pi_star[s1]))

    #print("You called Linear Programming")
else:
    print("Invalid algorithm given as input")