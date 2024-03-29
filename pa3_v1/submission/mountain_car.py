'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.num_discrete = 10 # along position axis & velocity axis, number of discrete vals taken 
        self.num_tilings = 10 #  8 .... 9 ... 10 ..... 11 .... 12
        self.epsilon_T1 = 0.1  
        self.epsilon_T2 = 0.07 # 0.07 pe -128.77 ..  0.09 pe -129.64
        self.learning_rate_T1 = 0.01
        self.learning_rate_T2 = 0.03 #0.08 pe -134.6, 0.07 pe -133.63 ,  0.04 pe -141.37 .... 0.04 pe -132.59  ... 0.03 pe -131.97, 0.02 pe -135.3 ...... 0.03 pe -141.69 ,0.02 pe -133.88  .... at 0.01  
        self.weights_T1 = np.zeros((3,1,self.num_discrete,self.num_discrete))
        self.weights_T2 = np.zeros((3,self.num_tilings,self.num_discrete,self.num_discrete))
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):
        # For Task-1
        feature_vector = np.zeros((3,1,self.num_discrete,self.num_discrete)) # (num_actions, num_tilings, num_pos, num_vel) 1 as 1 tiling
        bin_vector_pos = np.linspace(self.lower_bounds[0],self.upper_bounds[0],num=self.num_discrete-1)
        bin_vector_vel = np.linspace(self.lower_bounds[1],self.upper_bounds[1],num=self.num_discrete-1)
        row_id = np.digitize(obs[0],bin_vector_pos)
        col_id = np.digitize(obs[1],bin_vector_vel)
        for i in range(3):
            feature_vector[i][0][row_id][col_id] = 1     
        return feature_vector

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''

    def get_better_features(self, obs):
        # For Task-2
        feature_vector = np.zeros((3,self.num_tilings,self.num_discrete,self.num_discrete)) # (num_actions, num_tilings, num_pos, num_vel) 1 as 1 tiling
        bin_vector_pos = np.linspace(self.lower_bounds[0],self.upper_bounds[0],num=self.num_discrete-1)
        bin_vector_vel = np.linspace(self.lower_bounds[1],self.upper_bounds[1],num=self.num_discrete-1)
        pos_range = self.upper_bounds[0]-self.lower_bounds[0]
        vel_range = self.upper_bounds[1]-self.lower_bounds[1]
        del_pos = pos_range/((self.num_tilings)*(self.num_discrete))
        del_vel = vel_range/((self.num_tilings)*(self.num_discrete))
        row_id = np.digitize(obs[0],bin_vector_pos)
        col_id = np.digitize(obs[1],bin_vector_vel)
        for j in range(self.num_tilings):
            for i in range(3):
                feature_vector[i][j][row_id][col_id] = 1 
            bin_vector_pos += del_pos
            bin_vector_vel += del_vel
            row_id = np.digitize(obs[0],bin_vector_pos)
            col_id = np.digitize(obs[1],bin_vector_vel)
        return feature_vector

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):
        a = 1
        if np.random.uniform(0,1.0) < epsilon:
            a = np.random.randint(0,3)
        else:
            Q_t = np.zeros(3)
            for i in range(3):
                Q_t[i] = np.sum(state[i]*weights[i])
            a = np.argmax(Q_t)
        return a

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
        #ntiles =  state.shape[1]
        Q_st_at = np.sum(state[action]*weights[action],axis=(1,2))
        Q_stn_atn = np.sum(new_state[new_action]*weights[new_action],axis=(1,2))
        Target_init = reward + (self.discount*Q_stn_atn) - Q_st_at
        Target = np.tile(Target_init[:,np.newaxis,np.newaxis],(1,self.num_discrete,self.num_discrete))
        weights[action] = weights[action] + state[action]*learning_rate*(Target)
        self.weights_T1 = weights 
        self.weights_T2 = weights 
        return weights

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
       help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task=args['task']
    train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0) 
    agent.env.action_space.seed(0)
    if(train):
        agent.train(task)
    else:
        print(agent.test(task))
