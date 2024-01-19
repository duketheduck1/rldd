import gym
import utility 
from gym import spaces
import numpy as np
import ast
import astor 

source = '''def f1():
    """test"""
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
'''

test = '''def f1():
    """test"""
    a = 0
    for _ in range(3):
        a += 1
    return a

def main():
    return f1()
'''

class DeltaDebuggingEnvironment(gym.Env):
    def __init__(self, source, test, max_steps):
        super(DeltaDebuggingEnvironment, self).__init__()
        self.tree = ast.parse(source)
        UTI = utility.UTILITY()
        self.tree = UTI.hashTree(self.tree)
        self.program = UTI.setStatements(self.tree)
        self.state = self.program
        self.max_steps = max_steps
        self.current_step = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Binary actions: apply or do not apply a delta
        self.observation_space = spaces.Discrete(len(source))

    def reset(self):
        # Reset the environment to the initial state
        self.current_step = 0
        return self.state
    
   

    def step(self, action):
        # Execute the given action, update state, and return new state, reward, done, and info
        self.current_step += 1
        done = self.current_step >= self.max_steps
        self.state = self.apply_action(action)
        reward = self.compute_reward()        
        self.state = self.take_action(action)
        return self.state, reward, done, {}
    

    def apply_action(self, action):
        # Apply the delta based on the action
        if action == 1:  # Apply the delta
            # Your logic to apply the delta to the state
            pass
        return self.state

    def compute_reward(self):
        # Define your reward logic based on the debugging objectives
        # Example: Reward for minimizing input size and isolating defects
        if self.state == None:
            return 0
        return -len(self.state)  # Negative length as a simple example

    def take_action(self, action):
        # Implement the logic to apply the action to the state
        return self.state


env = DeltaDebuggingEnvironment(source, test, max_steps=100)
print(env.state)
state = env.reset()

for _ in range(10):
    action = env.action_space.sample()  # Sample a random action for testing
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Next State: {next_state}")
