import json
import pickle
import os
import tensorflow as tf
import csv
import os
class datasave():

    def save_model_parameters(self, model, filepath, rewards, state, scores):
        """
        Save the parameters of the RL agent's model to disk.

        Args:
        - model: The RL agent's model to save.
        - filepath: The filepath to save the model parameters.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(model.get_parameters(rewards, state, scores), file)

    def load_model_parameters(self, filepath):
        """
        Load the parameters of the RL agent's model from disk.

        Args:
        - filepath: The filepath to load the model parameters from.

        Returns:
        - model_parameters: The loaded model parameters.
        """
        with open(filepath, 'rb') as file:
            model_parameters = pickle.load(file)
        return model_parameters


    def save_training_data(self, env, filepath):
        """
        Save the training data to disk.

        Args:
        - training_data: The training data to save.
        - filepath: The filepath to save the training data.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(env.get_parameters(), file)

    def load_training_data(self, filepath):
        """
        Load the training data from disk.

        Args:
        - filepath: The filepath to load the training data from.

        Returns:
        - training_data: The loaded training data.
        """
        with open(filepath, 'rb') as file:
            training_data = pickle.load(file)
        return training_data



    def save_training_logs(self, training_logs, filepath):
        """
        Save the training logs to disk.

        Args:
        - training_logs: The training logs to save.
        - filepath: The filepath to save the training logs.
        """
        with open(filepath, 'w') as file:
            json.dump(training_logs, file)

    def load_training_logs(self, filepath):
        """
        Load the training logs from disk.

        Args:
        - filepath: The filepath to load the training logs from.

        Returns:
        - training_logs: The loaded training logs.
        """
        with open(filepath, 'r') as file:
            training_logs = json.load(file)
        return training_logs


    def save_checkpoint(self, model, optimizer, epoch, filepath):
        """
        Save model checkpoint to disk.

        Args:
        - model: The trained model.
        - optimizer: The optimizer used for training.
        - epoch: The current training epoch.
        - filepath: The filepath to save the checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        tf.saved_model.save(checkpoint, filepath)

    def load_checkpoint(self, model, optimizer, filepath):
        """
        Load model checkpoint from disk.

        Args:
        - model: The model to load the checkpoint into.
        - optimizer: The optimizer to load the checkpoint into.
        - filepath: The filepath of the checkpoint to load.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")
    
        checkpoint = tf.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return model, optimizer, epoch



class CustomObject:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def to_json(self):
        return json.dumps({'name': self.name, 'age': self.age})

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(data['name'], data['age'])

# Example usage
##obj = CustomObject('Alice', 30)
##serialized_data = obj.to_json()
##print(serialized_data)  # Output: '{"name": "Alice", "age": 30}'
##deserialized_obj = CustomObject.from_json(serialized_data)
##print(deserialized_obj.name, deserialized_obj.age)  # Output: Alice 30

def serialize_config(config, filename):
    """Serialize configuration parameters and save to a file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def deserialize_config(filename):
    """Load serialized configuration parameters from a file."""
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

# Example configuration parameters
config = {
    'model_type': 'CNN',
    'num_layers': 3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'data_path': '/path/to/data',
    'output_path': '/path/to/output',
}

# Save configuration parameters to a file
##serialize_config(config, 'config.json')

# Load configuration parameters from the saved file
##loaded_config = deserialize_config('config.json')
##print(loaded_config)

def train_rl_agent_with_saving(agent, environment, episodes=1000, save_interval=100):
    for episode in range(1, episodes + 1):
        # Perform RL training steps
        
        # Save model parameters, logs, etc. at specified intervals
        if episode % save_interval == 0:
            save_model_parameters(agent, episode)
            #save_training_logs(logs, episode)
            save_checkpoint(agent, episode)
            save_configuration(config)



def save_model_parameters(model, episode):
    # Define the directory and file name for saving
    save_dir = 'saved_models/'
    file_name = f'model_parameters_episode_{episode}.pt'
    save_path = save_dir + file_name

    # Serialize and save the model parameters
    try:
        tf.saved_model.save(model.state_dict(), save_path)
        print(f"Model parameters saved successfully at {save_path}")
    except Exception as e:
        print(f"Error saving model parameters: {e}")
#-------------
def save_training_logs(logs, log_file):
    # Define the field names for the CSV file
    field_names = ['Episode', 'Reward', 'Loss']

    # Write the logs to the CSV file
    try:
        with open(log_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            # Write header only if the file is empty
            if file.tell() == 0:
                writer.writeheader()
            # Write logs for each episode
            for log in logs:
                writer.writerow(log)
        print(f"Training logs saved successfully at {log_file}")
    except Exception as e:
        print(f"Error saving training logs: {e}")
#-------------
def save_checkpoint(checkpoint, checkpoint_manager, checkpoint_prefix):
    # Save the checkpoint
    checkpoint_manager.save(checkpoint_number=checkpoint_prefix)
    print(f"Checkpoint saved at step {checkpoint_prefix}")

# Example usage
##checkpoint_dir = './checkpoints'
##checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
##checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
##checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# Inside the training loop
##for step in range(num_steps):
    # Perform training step
    ##train_step()
    
    # Save checkpoint every n steps
    ##if (step + 1) % checkpoint_interval == 0:
        ##save_checkpoint(checkpoint, checkpoint_manager, step + 1)
#-----------

def save_configuration(config, filepath):
    # Serialize configuration object to JSON
    serialized_config = json.dumps(config, indent=4)
    
    # Write serialized configuration to file
    with open(filepath, 'w') as file:
        file.write(serialized_config)

# Example usage
##config = {
    ##'learning_rate': 0.001,
    ##'batch_size': 32,
    ##'num_episodes': 1000,
    # Add other configuration parameters as needed
##}
##config_file = 'config.json'
##save_configuration(config, config_file)
#------

def save_optimizer_state(optimizer, filepath):
    # Get the optimizer's state as a dictionary
    optimizer_state = optimizer.get_weights()
    
    # Save the optimizer's state to a file
    with open(filepath, 'wb') as f:
        pickle.dump(optimizer_state, f)

def load_optimizer_state(optimizer, filepath):
    # Load the optimizer's state from a file
    with open(filepath, 'rb') as f:
        optimizer_state = pickle.load(f)
    
    # Set the optimizer's state
    optimizer.set_weights(optimizer_state)
#------
import json

def save_training_config(config, filepath):
    # Save the configuration parameters to a JSON file
    with open(filepath, 'w') as f:
        json.dump(config, f)

def load_training_config(filepath):
    # Load the configuration parameters from a JSON file
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config
#------
import csv

def save_training_history(history, filepath):
    # Save the training history to a CSV file
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = history[0].keys()  # Assuming history is a list of dictionaries
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write each row of the training history
        for entry in history:
            writer.writerow(entry)

def load_training_history(filepath):
    # Load the training history from a CSV file
    history = []
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            history.append(dict(row))
    return history


#-----------
"""

def save_model_architecture(model, filepath):
    # Save the model architecture to a SavedModel format
    tf.saved_model.save(model, filepath)
#------
import json

def save_environment_configuration(config, filepath):
    # Save the environment configuration to a JSON file
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

environment_config = {
    'state_space': ...,  # Details about the state space
    'action_space': ...,  # Details about the action space
    'reward_function': ...,  # Details about the reward function
    # Other parameters and settings
}

save_environment_configuration(environment_config, 'environment_config.json')

def load_environment_configuration(filepath):
    # Load the environment configuration from a JSON file
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config
#----------
metadata = {
    "timestamp": "YYYY-MM-DD HH:MM:SS",  # Timestamp indicating when the model was trained
    "author": "Your Name",  # Authorship information
    "description": "Trained model for solving Delta Debugging problem",  # Description of the problem domain
    "model_version": "1.0",  # Version identifier for the trained model
    "training_duration": "XX hours YY minutes ZZ seconds",  # Duration of the training process
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "Adam",
        # Add other hyperparameters as needed
    },
    "training_metrics": {
        "final_loss": 0.123,
        "final_accuracy": 0.95,
        # Add other relevant metrics obtained during training
    },
    "dataset_info": {
        "name": "Delta Debugging Dataset",
        "source": "Internal database",
        "size": "1000 samples",
        "features": ["feature1", "feature2", ...],
        "labels": ["label1", "label2", ...],
        # Add other dataset information
    },
    "environment_details": {
        "state_space": "Continuous",
        "action_space": "Discrete",
        "reward_function": "Custom",
        # Add other details about the RL environment
    },
    "software_dependencies": {
        "python": "3.8",
        "tensorflow": "2.6.0",
        "numpy": "1.21.2",
        # Add versions of other libraries/frameworks
    },
    "hardware_specifications": {
        "cpu": "Intel Core i7-10700K",
        "gpu": "NVIDIA GeForce RTX 3080",
        # Add details about the hardware environment
    }
}
"""

def load_model_parameters(filepath):
        """
        Load the parameters of the RL agent's model from disk.

        Args:
        - filepath: The filepath to load the model parameters from.

        Returns:
        - model_parameters: The loaded model parameters.
        """
        with open(filepath, 'rb') as file:
            model_parameters = pickle.load(file)
        return model_parameters