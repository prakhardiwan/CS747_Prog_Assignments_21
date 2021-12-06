import argparse
import random
import numpy as np
import math


# Function Definitions

############ Main Program #################################################################################

# Extracting Inputs 
parser = argparse.ArgumentParser(description='Taking inputs')
parser.add_argument("--value-policy", help="Value-Policy File Path", type=str, default="value_and_policy_file")
parser.add_argument("--states", help="State File Path", type=str, default="data/attt/states/states_file_p1.txt")
parser.add_argument("--player-id", help = "Agent ID", type = int, default=1)

inp = parser.parse_args()
State_map = []
# Reading the states of agent
with open(inp.states) as text_data:
    line_read = text_data.readlines()      # line_read is the lines read from text file
#print(line_read)
num_States = len(line_read)

#print(num_States)
for i in range(len(line_read)):
    line_r = line_read[i].split()
    State_map.append(line_r[0])
    #print(State_map[i])

print(inp.player_id) 
# Reading the policy file
with open(inp.value_policy) as text_vp_data:
    vp_line_read = text_vp_data.readlines()      
#print(len(vp_line_read)-2)

for i in range(len(vp_line_read)-2):    
    line_split = vp_line_read[i].split()
    V_s = float(line_split[0])
    pi_s = int(line_split[1])
    if V_s==0 :    # in this case we need to pick just a legal action since Vpi is 0 
        state = State_map[i]
        possible_actions = []
        result_array = np.zeros((9,),dtype=int)
        for k in range(9):
            if state[k] == "0":
                possible_actions.append(k)
                result_array[k] = 1
                break
        print("{} {} {} {} {} {} {} {} {} {}".format(State_map[i],result_array[0],result_array[1],result_array[2],result_array[3],result_array[4],result_array[5],result_array[6],result_array[7],result_array[8]))
        #if(len(possible_actions)==0):
        #    print("Why")
    else:
        result_array = np.zeros((9,),dtype=int)
        result_array[pi_s] = 1
        print("{} {} {} {} {} {} {} {} {} {}".format(State_map[i],result_array[0],result_array[1],result_array[2],result_array[3],result_array[4],result_array[5],result_array[6],result_array[7],result_array[8]))
