import gym
import random
import numpy as np
from tensorflow import keras
from keras import layers
import makeData
import matplotlib.pyplot as plt
from datasave import datasave
from copy import deepcopy
from time import time

# Custom Gym environment for Delta Debugging
class DeltaDebuggingEnv(gym.Env):
    def __init__(self, data, target):
        super(DeltaDebuggingEnv, self).__init__()
        #data = (dataTrain.tree, dataTrain.program, dataTrain.codeprogram)
        self.data = data
        self.target = target
        self.tree = self.data.tree
        self.program = [1 for i in range(len(self.data.program))]#self.data.program
        self.current_state = self.program
        self.target_state = target.code        
        self.action_space = gym.spaces.Discrete(len(self.program))
        self.observation_space = gym.spaces.Discrete(len(self.program))
        self.max_steps = 100  # Maximum steps for each episode        
        self.cothethat = [False,False]
        self.noidung = ""        

    def step(self, action):
        old_state = [i for i in self.current_state]        
        self.current_state = self.reduce_state(self.current_state, action)
        reward = self.calculate_reward(old_state, self.current_state)
        done = self.is_done(self.current_state) or (self.max_steps <= 0)       
        self.max_steps -= 1
        return self.current_state, reward, done, {}

    def reduce_state(self, state, action):
        if action < len(state):
            state[action] = 0
        return state

    def calculate_reward(self, old_state, new_state):
        lentarget = len(self.target.code)        
        lencurrentstate = len(self.data.convertto(new_state))
        lenold_state = len(self.data.convertto(old_state))       
        if lencurrentstate < lenold_state and lencurrentstate > lentarget:
            reward = lentarget/lencurrentstate*10 
        elif lencurrentstate < lentarget:
            reward = (lencurrentstate-lentarget)/lentarget*10
        elif lencurrentstate == lentarget:            
            reward = 100
        else:
            reward = 0
        return reward
  
    def is_done(self, state):
        tempcode = self.data.convertto(state)        
        done2 = np.array_equal(tempcode, self.target.code)
        done3 = len(tempcode) < len(self.target.code)
        done = len(tempcode) == len(self.target.code)        
        return done3 or done or state == []

    def reset(self):
        self.current_state =  [1 for i in self.program]
        self.max_steps = 100
        return self.current_state
    def get_parameters(self):
        results = {
            "program": self.program,
            "current_state": self.current_state,
            "target_state": self.target_state,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "max_steps":self.max_steps       
        }
        return results

# DQN Agent
class DQNAgent:
    def __init__(self, env, epsilon = 0.19, gamma = 0.75):
        self.state_dim = len(env.program)
        self.action_dim = self.state_dim
        self.model = self.build_model()
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = self.state_dim #batch_size
        self.batch_size = self.state_dim
        self.memory = []        
        self.datacheck = []

    def build_model(self):
        model = keras.Sequential([            
            layers.Dense(24, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, len(state)-1)
        else:            
            q_values = self.model.predict([state])            
            emin =  np.min(q_values) - 1
            emax =  np.max(q_values) +1
            action = np.argmax([q_values[0][i] if state[i] else emin for i in range(len(state))])
            #action = np.argmin([q_values[0][i] if state[i] else emax for i in range(len(state))])
            '''a=[1,2,3,4,5]
            b=[0,1,0,0,1]
            amax=max(a)+1
            e = [ a[i] if b[i] else amax for i in range(len(a))]'''
                   
        return action

    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))        
        if len(self.memory) >= self.batch_size and len(next_state) > 0:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            target_q_values = self.calculate_target_q_values(batch)
            self.model.fit(np.array(states), target_q_values, epochs=1, verbose=0)
            if(np.max(target_q_values) > 0):
                return(np.sum(target_q_values/np.max(target_q_values)*100))        
        return (0)

    def calculate_target_q_values(self, batch):        
        target_q_values = []
        for state, action, reward, next_state, done in batch:            
            q_values = self.model.predict([state])            
            next_q_values = self.model.predict([next_state])
            if done:
                q_values[0][action] = reward
            else:
                q_values[0][action] = reward + self.gamma * np.max(next_q_values)
            target_q_values.append(q_values[0])        
        return target_q_values
    
    def get_parameters(self):
        results = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'model': self.model,
            'epsilon': self.epsilon,
            'num_actions': self.num_actions,
            'batch_size': self.batch_size,
            'memory': self.memory,            
            'datacheck': self.datacheck
        }
        return results
        
def plot_running_avg(totalrewards, title):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
	plt.plot(running_avg)
	plt.title("Running "+title+ " Average")
	plt.show()
# Constants

def train_agent(env, agent, enpisodes): 
    
    rewards=[]
    scores = []
    for episode in range(enpisodes ):
        state = env.reset() 
        done = False
        total_reward = []
        avgscore = []        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)            
            
            total_reward.append(reward)
            score = agent.train(state, action, reward, next_state, done)
            avgscore.append(score)
            if np.sum(next_state) <= 1 or np.sum(state) <= 2:
                break
            state = [i for i in next_state]
            
        avg_reward = sum(total_reward)/len(total_reward)
        rewards.append(avg_reward)
        scores.append(sum(avgscore)/len(avgscore))
        print(f"Episode {episode}, AvgReward: {avg_reward}")
    return rewards, state, scores

def test_agent(env, agent, enpisodes): 
    
    rewards=[]
    
    for episode in range(enpisodes ):
        state = env.reset() 
        done = False
        total_reward = []         
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)            
            
            total_reward.append(reward)
            agent.train(state, action, reward, next_state, done)
            if np.sum(next_state) <= 1 or np.sum(state) <= 2:
                break
            state = [i for i in next_state]
            
        avg_reward = sum(total_reward)/len(total_reward)
        rewards.append(avg_reward)
        print(f"Episode {episode}, AvgReward: {avg_reward}")
    return rewards, state
    #-------

def run(problematic_program, target_state, test_code):
    start = time()
    data = deepcopy(makeData.convertData(problematic_program))
    datatg = deepcopy(makeData.convertData(target_state))    
    env = DeltaDebuggingEnv(data, datatg)    
    agent = DQNAgent(env, epsilon = 0.19, gamma = 0.75)    
    
    rewards, state, scores = train_agent(env, agent, 1000)
    
    print("Pro 1:",state)    
    temp = state
    print(data.convertto(temp))
    saveData = datasave()
    saveData.save_model_parameters(agent, "_agent.txt")
    saveData.save_training_data(env, "_train.txt")    
    print("End agent")
    
    dataTest = makeData.convertData(test_code)
    treep = dataTest.tree    
    test_program = dataTest.program
    """
    print("Starting Testing")
    state = test_program 
    print(state)
    env.current_state = state
    print(env.current_state)    
    while True:
        action = agent.select_action(state)
        next_state, _, done, _ = env.step(action)        
        if len(next_state) <= 1 or len(state) <= 2:        
                break
        state = [i for i in next_state]    

    # Decode the minimal program state back to a string
    minimal_program = state
    print("Minimal program:", minimal_program)    
    
    temp = minimal_program
    print("agent", len(agent.memory),temp)
    print(dataTest.convertto(temp))
    """
    end = time()-start
    print(f"Duration: {end}")
    plot_running_avg(rewards, "Rewards")
    plot_running_avg(scores,"Scores") 
    print("agent", len(agent.memory),temp)   
if __name__ == "__main__":
    problematic_program = """
def f1():
    '''Test'''
    a = 0
    for _ in range(3):
        a += 1
    return a

def f2():
    print('Woot?')


def f3():
    print('42 and you know it')


def main():
    return f1()
"""
    target_state = """
def f1():
    '''Test'''
    a = 0
    for _ in range(3):
        a += 1
    return a

def main():
    return f1()
"""
    test_code = """
def f1():
    '''Test'''
    a = 0
    for _ in range(3):
        a += 1
    return a

def f2():
    print('Woot?')


def f3():
    print('42 and you know it')


def main():
    return f1()
"""
run(problematic_program, target_state, test_code)
    