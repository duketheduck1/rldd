import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Custom Gym environment for Delta Debugging
class DeltaDebuggingEnv(gym.Env):
    def __init__(self, program, target_state):
        super(DeltaDebuggingEnv, self).__init__()
        self.program = program
        self.current_state = self.encode_state(program)
        self.target_state = self.encode_state(target_state)#np.zeros_like(self.current_state)  # Set this to the minimal version that reproduces the issue
        print("self.target_state: ", self.target_state)
        self.action_space = gym.spaces.Discrete(len(program))
        self.observation_space = gym.spaces.Discrete(len(program))
        self.max_steps = 100  # Maximum steps for each episode

    def encode_state(self, state):
        # Encode the state as a numerical array
        return np.array(list(map(ord, state)))

    def step(self, action):
        self.current_state = self.reduce_state(self.current_state, action)
        reward = -abs(len(self.current_state) - len(self.target_state))
        done = (np.array_equal(self.current_state, self.target_state)) or (self.max_steps <= 0)
        self.max_steps -= 1
        return self.current_state, reward, done, {}

    def reduce_state(self, state, action):
        # Apply the action to reduce the current state
        state = np.delete(state, action)
        return state

    def reset(self):
        self.current_state = self.encode_state(self.program)
        self.max_steps = 100
        return self.current_state

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.build_model()
        self.epsilon = 0.1
        self.num_actions = 10
        self.memory = []

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
            action = random.randint(0, self.num_actions - 1)
        else:            
            q_values = 0.1#self.model.predict(np.array([state]))
            action = np.argmax(q_values)
        return action

    def train(self, state, action, reward, next_state, done):
        self.memory.append((sum(state)/1000, action, reward, sum(next_state)/1000, done))
        print("state:", ''.join(map(chr, state)))
        print("next_state:",''.join(map(chr, next_state)))
        if len(self.memory) >= batch_size:
            batch = np.array(self.memory)
            states = np.vstack(batch[:, 0])
            actions = np.array(batch[:, 1], dtype=int)
            rewards = np.array(batch[:, 2], dtype=float)
            next_states = np.vstack(batch[:, 3])
            dones = np.array(batch[:, 1], dtype=bool)              
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
    problematic_program = """
    def divide(a, b):
        result = a / b
        return result

    result = divide(10, 0)
    print(result)
    """
    target_state = """
    def divide(a, b):
        result = a / b
        return result        
    """
    env = DeltaDebuggingEnv(problematic_program, target_state)

    state_dim = len(problematic_program)
    action_dim = state_dim

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
            state = next_state

        print(f"Episode {episode}, Total Reward: {total_reward}") 

    test_program = """
    def divide(a, b):
        result = a / b
        return result
        khong ai ca
    """
    state = env.encode_state(test_program)
    print(state)
    while True:
        action = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        print(next_state)
        state = next_state
        if done:
            break
        
       

    # Decode the minimal program state back to a string
    minimal_program = ''.join(map(chr, state))
    print("Minimal program:", minimal_program)

if __name__ == "__main__":
    main()