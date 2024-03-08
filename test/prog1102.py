import gym
import random
import ast
import astor 
import utility
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Custom Gym environment for Delta Debugging
class DeltaDebuggingEnv(gym.Env):
    def __init__(self, source, target_state):
        super(DeltaDebuggingEnv, self).__init__()
        self.UTI = utility.UTILITY()
        self.tree, self.program = self.encode_code(source)# Encode the source code
        self.current_state = self.program
        self.tree_target, self.target_state = self.encode_code(target_state)#np.zeros_like(self.current_state)  # Set this to the minimal version (target state) that reproduces the issue
        self.action_space = gym.spaces.Discrete(len(self.program))
        self.observation_space = gym.spaces.Discrete(len(self.program))
        self.max_steps = 100  # Maximum steps for each episode

    # Encode the given code 
    def encode_code(self, code):
        tree = ast.parse(code)
        tree = self.UTI.hashTree(tree)
        program = self.UTI.setStatements(tree)
        return tree, program

    # Take a step in the environment of Q learning
    def step(self, action):
        self.current_state = self.reduce_state(self.current_state, action) # Reduce the current state
        reward = -abs(len(self.current_state) - len(self.target_state)) # Calculate the reward
        done = (np.array_equal(self.current_state, self.target_state)) or (self.max_steps <= 0) # Check if episode is done
        self.max_steps -= 1
        return self.current_state, reward, done, {}  # Return the new state, reward, done flag, and additional info

    # Reduce the current state based on the given action
    def reduce_state(self, state, action):
        if len(state) > action:
            state = np.delete(state, action)
        return state

    # Reset the environment to its initial state
    def reset(self):
        self.current_state = self.program
        self.max_steps = 100
        return self.current_state

# DQN Agent
class DQNAgent:
    # Initialize the DQN Agent with the given state and action dimensions
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.build_model() # Build the DQN model
        self.epsilon = 0.1
        self.num_actions = 10
        self.memory = [] # Initialize the replay memory

    #build the model
    def build_model(self):
        model = keras.Sequential([             
            layers.Dense(24, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    # Select an action based on the current state
    def select_action(self, state):
        # if random.random() < self.epsilon:
        #     action = random.randint(0, self.num_actions - 1)
        # else:            
        #     q_values = 0.1#self.model.predict(np.array([state]))
        #     action = np.argmax(q_values)
        action = random.randint(0, len(state) )
        return action

    # Train the DQN Agent based on the experience tuple
    def train(self, state, action, reward, next_state, done):
        # Store the experience in the replay memory
        self.memory.append((state, sum(state)/1000, action, reward, next_state, sum(next_state)/1000, done))
        # Perform a gradient descent step
        print("state:", state)
        print("next_state:",next_state)
        if len(self.memory) >= batch_size:
            batch = np.array(self.memory)
            states = np.vstack(batch[:, 1])
            actions = np.array(batch[:, 2], dtype=int)
            rewards = np.array(batch[:, 3], dtype=float)
            next_states = np.vstack(batch[:, 5])
            dones = np.array(batch[:, 6], dtype=bool)              
            q_values = self.model.predict(states)
            next_q_values = self.model.predict(np.array(next_states))
            target_q_values = q_values.copy()

            for i in range(batch_size):
                if dones[i]:
                    target_q_values[i, actions[i]] = rewards[i]
                else:
                    target_q_values[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])

            self.model.fit(states, target_q_values, epochs=1, verbose=0)

# Constants
epsilon = 0.1
gamma = 0.99
batch_size = 32

def main():
    # Define the problematic program and the target state
    problematic_program = '''def f1():
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
    target_state = '''def f1():
    """test"""
    a = 0
    for _ in range(3):
        a += 1
    return a

    def main():
        return f1()
    '''
    # Create the environment
    env = DeltaDebuggingEnv(problematic_program, target_state)

    state_dim = len(problematic_program)
    action_dim = state_dim

    # Create the DQN agent
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(5):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.train(state, action, reward, next_state, done)
            if len(state) <= 0 or len(next_state) <= 0:
                break
            state = next_state

        print(f"Episode {episode}, Total Reward: {total_reward}") 

    test_program = '''def f1():
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
    tree_test, state = env.encode_code(test_program)
    env.current_state = state
    print(state)
    while True:
        action = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        print(next_state)
        
        if done:
            break
        if len(state) <= 0 or len(next_state) <= 0:
                break
        state = next_state
       

    # Decode the minimal program state back to a string
    minimal_program = state
    print("Minimal program:", minimal_program)

if __name__ == "__main__":
    main()