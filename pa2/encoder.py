import argparse
import random
import numpy as np
import math
# Function Definitions
def in_rows(s2,opponent_ID):
    if(s2[0]==str(opponent_ID)) and (s2[1]==str(opponent_ID)) and (s2[2]==str(opponent_ID)):
      return 1
    elif(s2[3]==str(opponent_ID)) and (s2[4]==str(opponent_ID)) and (s2[5]==str(opponent_ID)):
      return 1
    elif(s2[6]==str(opponent_ID)) and (s2[7]==str(opponent_ID)) and (s2[8]==str(opponent_ID)):
      return 1
    return 0
def in_cols(s2,opponent_ID):
    if(s2[0]==str(opponent_ID)) and (s2[3]==str(opponent_ID)) and (s2[6]==str(opponent_ID)):
      return 1
    elif(s2[1]==str(opponent_ID)) and (s2[4]==str(opponent_ID)) and (s2[7]==str(opponent_ID)):
      return 1
    elif(s2[2]==str(opponent_ID)) and (s2[5]==str(opponent_ID)) and (s2[8]==str(opponent_ID)):
      return 1
    return 0
def in_diag(s2,opponent_ID):
    if(s2[0]==str(opponent_ID)) and (s2[4]==str(opponent_ID)) and (s2[8]==str(opponent_ID)):
      return 1
    elif(s2[2]==str(opponent_ID)) and (s2[4]==str(opponent_ID)) and (s2[6]==str(opponent_ID)):
      return 1
    return 0
    
def check_if_winner(s2,opponent_ID):
    if in_rows(s2,opponent_ID)==1:
        return 1
    elif in_cols(s2,opponent_ID)==1:
        return 1
    elif in_diag(s2,opponent_ID)==1:
        return 1
    return 0

############ Main Program #################################################################################

# Extracting Inputs 
parser = argparse.ArgumentParser(description='Taking inputs')
parser.add_argument("--policy", help="Policy File Path", type=str, default="data/attt/policies/p2_policy2.txt")
parser.add_argument("--states", help="State File Path", type=str, default="data/attt/states/states_file_p1.txt")

inp = parser.parse_args()
# Reading the states of agent
with open(inp.states) as text_data:
    line_read = text_data.readlines()      # line_read is the lines read from text file
num_states = len(line_read)
num_actions = 9
State_Map = []
Tran_Prob = np.zeros((num_states,num_actions,num_states),dtype=np.float32)
Reward = np.zeros((num_states,num_actions,num_states),dtype=np.int8)
#print(num_states)
for i in range(len(line_read)):
    line_r = line_read[i].split()
    State_Map.append(line_r[0])
#print(State_Map[0])
#print(State_Map[1])
print("numStates",(num_states+2))
print("numActions",9)
# Terminal state placeholders
rewarding_terminal_S = "222222222"  ## no actions can be done on this as per code below so will just pass through
unrewarding_terminal_S = "111111111" ## no actions can be done on this as per code below so will just pass through
State_Map.append(rewarding_terminal_S)
State_Map.append(unrewarding_terminal_S)
print("end",State_Map.index(unrewarding_terminal_S),State_Map.index(rewarding_terminal_S))
## S1 A S2 T(S1,A,S2) R(S1,A,S2)
with open(inp.policy) as text_data:
    po_line_read = text_data.readlines()
opponent_ID = 0
line_0 = po_line_read[0].split()
if line_0[0]=="1":
    opponent_ID = 1
elif line_0[0]=="2":
    opponent_ID = 2
#print(opponent_ID)
if opponent_ID == 2:
    agent_ID = 1
elif opponent_ID == 1:
    agent_ID = 2
#print(agent_ID)    
num_op_states = len(po_line_read)-1
opponent_policy_data = {}

## Filling up the terminal states
S_terminal_agent_side = []
S_terminal_opponent_side = []
S_inter = [] #intermediate state
PA_env = []

for i in range(1,len(po_line_read)):
    po_line_r = po_line_read[i].split()
    S_inter.append(po_line_r[0])
    for j in range(1,10):
        PA_env.append(po_line_r[j])
    
possible_actions = [] # actions by the agent 
possible_actions_env = [] # actions by environment
for s1 in State_Map:
    num_s1 = State_Map.index(s1)
    possible_actions = []
    for i in range(9):
        if s1[i] == "0":
            possible_actions.append(i)
    for action in possible_actions:
        state = ""
        list_s1 = list(s1)
        list_s1[action] = agent_ID
        for k in range(9):
            state += str(list_s1[k])
        if(S_inter.count(state)==0): # reward remains 0 i.e in policy file state isn't there 
            S_terminal_opponent_side.append(state)
            #print("Terminal at opponent side")
            ## maybe give negative rewards
            s2 = unrewarding_terminal_S  # for types of terminal states which lead to no reward
            num_s2 = State_Map.index(s2)
            print("transition",num_s1,action,num_s2,0,1)
        else:                       
            S_inter_number = S_inter.index(state)
            possible_actions_env = [] 
            for di in range(9):
                if state[di] == "0":
                    possible_actions_env.append(di)
            for h in possible_actions_env:
                s2 = ""
                list_s_inter = list(state)
                list_s_inter[h] = opponent_ID
                for u in range(9):
                    s2 += str(list_s_inter[u])
                if(State_Map.count(s2)==0):
                    #print("Terminal state at agent side: so check for rewards here")
                    ## reward 1 can happen here only
                    index_prob = h+(S_inter_number*9)
                    winner = check_if_winner(s2,opponent_ID)
                    if winner == 1:
                        s2 = rewarding_terminal_S   # for types of terminal states which lead to reward
                        num_s2 = State_Map.index(s2)
                        print("transition",num_s1,action,num_s2,1,PA_env[index_prob])
                    else:
                        s2 = unrewarding_terminal_S  # for types of terminal states which lead to no reward
                        num_s2 = State_Map.index(s2)
                        print("transition",num_s1,action,num_s2,0,PA_env[index_prob])
                else:
                    index_prob = h+(S_inter_number*9)
                    num_s2 = State_Map.index(s2)
                    print("transition",num_s1,action,num_s2,0,PA_env[index_prob])

gamma = 1
print("mdptype episodic")
print("discount ",gamma)
