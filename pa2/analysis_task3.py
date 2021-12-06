import argparse
import random
import numpy as np
import math

# All comments are based on the fact the pi_0 was for player2 as mentioned in the task3 description, though you upon making edits can run for pi_0 being for player1
# This file is used for generating the difference in number of actions of intermediate policies from the last policy for player 1 and 2 which are pi_19.txt and pi_20.txt
last_policy = "pi_20.txt" # in case of player 1 policies change to pi_19.txt, in case of player 2 policies change to pi_20.txt

with open(last_policy) as text_data:
    po_line_read = text_data.readlines()
action_last_policy = np.zeros(len(po_line_read)-1)
for i in range(1,len(po_line_read)):
    po_line_r = po_line_read[i].split()
    state = po_line_r[0]
    for j in range(1,10):
        if float(po_line_r[j]) == 1.0: # works only for deterministic policies which indeed are outputted from decoder i.e. pi[1], pi[2]  and so on.. only pi[0] can be both deterministic and stochastic(is ignored)
                action_last_policy[i-1] = j-1

error = []
policy = "pi_"
for k in range(1,10):  # in case of player 1 policies change to range(10), in case of player 2 policies change to range(1,10)
    policy = "pi_"
    policy += str(2*k)+".txt"  # in case of player 1 policies change to 2k+1, in case of player 2 policies change to 2k
    with open(policy) as text_data:
        po_line_read = text_data.readlines()
    last_policy_1 = np.zeros(len(po_line_read)-1)
    for i in range(1,len(po_line_read)):
        po_line_r = po_line_read[i].split()
        state = po_line_r[0]
        for j in range(1,10):
            if float(po_line_r[j]) == 1.0: # works only for deterministic policies which indeed are outputted from decoder i.e. pi[1], pi[2]  and so on.. only pi[0] can be both deterministic and stochastic(is ignored)
                last_policy_1[i-1] = j-1
    diff = 0
    for r in range(len(po_line_read)-1):
        if last_policy_1[r]!=action_last_policy[r]:
            diff +=1
    error.append(diff) # stores errors in policies in an increasing order of time
    print("Difference between pi_{} and pi_20 policies is at {} number of states".format(2*k,diff))  # in case of player 1 use 2k+1, in case of player 2 use 2k