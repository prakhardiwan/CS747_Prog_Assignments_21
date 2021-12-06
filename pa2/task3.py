import os 

# Here I initially used p2_policy1.txt as an input policy i.e. pi[0] as per the question statement
# so it's copy is saved as pi_0.txt file (to be used for analysis of Player 2's policy convergence)
# and it's stored alongside the planner.py and other files
# Player 2 policy files : pi_0.txt (initial inputted policy), pi_2.txt, pi_4.txt till .. pi_20.txt 
# Player 2 policy files : pi_1.txt, pi_3.txt till .. pi_19.txt
# Have chosen p2_policy1 as it's a deterministic policy and easier for comparision hence with future policies of P2
# I assume that files are as in the zipped directory provided

# Even though other 3 combinations can be comfortably run upon editing this file appropriately
mdpfile = "mdpfile"
v_and_p_file = "v_and_p_file"
for i in range(10): 
    policy_file_path_P2 = "pi_"
    policy_file_path_P2 += str(2*i) + ".txt"
    policy_file_path_P1 = "pi_"
    policy_file_path_P1 += str((2*i)+1) + ".txt"
    os.system("python encoder.py --policy {} --states data/attt/states/states_file_p1.txt > {}".format(policy_file_path_P2,mdpfile+str(2*i)))
    os.system("python planner.py --mdp {} > {}".format(mdpfile+str(2*i),v_and_p_file+str(2*i)))
    os.system("python decoder.py --value-policy {} --states data/attt/states/states_file_p1.txt --player-id 1 > {}".format(v_and_p_file+str(2*i),policy_file_path_P1))

    policy_file_path_P2 = "pi_"
    policy_file_path_P2 += str(2*i+2) + ".txt"
    os.system("python encoder.py --policy {} --states data/attt/states/states_file_p2.txt > {}".format(policy_file_path_P1,mdpfile+str((2*i)+1)))
    os.system("python planner.py --mdp {} > {}".format(mdpfile+str((2*i)+1),v_and_p_file+str((2*i)+1)))
    os.system("python decoder.py --value-policy {} --states data/attt/states/states_file_p2.txt --player-id 2 > {}".format(v_and_p_file+str((2*i)+1),policy_file_path_P2))
    print(i)