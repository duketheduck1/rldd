import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define the Delta Debugging environment
class DeltaDebuggingEnv(gym.Env):
    def __init__(self, problematic_inputs, target_inputs):
        super(DeltaDebuggingEnv, self).__init__()
        self.problematic_inputs = problematic_inputs
        self.target_inputs = target_inputs
        self.current_input = None
        self.episode_step = 0
        self.max_steps = len(str(problematic_inputs[0])) # Assuming all inputs have the same length
        self.action_space = gym.spaces.Discrete(len(str(problematic_inputs[0])))
        self.observation_space = gym.spaces.Discrete(len(str(problematic_inputs[0])))

    def reset(self):
        self.current_input = np.copy(self.problematic_inputs)
        self.episode_step = 0
        return self.current_input

    def step(self, action):
        # Apply the selected action (e.g., delete a line of code)       
        self.current_input[action]= self.target_inputs[action]
        self.episode_step += 1
        done = self.episode_step >= self.max_steps
        reward = -1 if not done else 0 # Penalize steps until done
        return self.current_input, reward, done, {}

# Define a simple neural network model
def create_model(input_shape, num_actions):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the reinforcement learning agent
class DeltaDebuggingAgent:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample() # Random action
        q_values = self.model.predict(state)
        return np.argmax(q_values)

    def train(self, num_episodes, gamma=0.99, epsilon=0.1):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                # Q-learning update
                target = reward + gamma * np.max(self.model.predict(next_state))
                target_f = self.model.predict(state)
                target_f[action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

                total_reward += reward
                state = next_state

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Load your dataset of problematic and target inputs here

problematic_inputs = np.array([8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
target_inputs = np.array([1, 2, 0, 0, 2, 0, 7, 8, 0, 10, 11])


# Create the Delta Debugging environment
env = DeltaDebuggingEnv(problematic_inputs, target_inputs)

# Create the neural network model
model = create_model(input_shape=env.observation_space.n, num_actions=env.action_space.n)

# Create the Delta Debugging agent
agent = DeltaDebuggingAgent(env, model)

# Train the agent
num_episodes = 100
agent.train(num_episodes)
print("da ra den day chƯa")
# Test the trained agent on problematic inputs
test_input = problematic_inputs
while True:
    action = agent.choose_action(test_input, epsilon=0.1) # Choose the best action
    test_input, _, done, _ = env.step(action)
    if done:
        break
minimized_input = test_input[0]

# Print the minimized input
print("Minimized Input:")
print(minimized_input)