import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from env import AggDelivState, PathInfo
from utils import MAX_PATHS, BATCH_SIZE, STATE_SIZE, ACTION_SIZE
from collections import deque
import os

# Hyperparameters

GAMMA = 0.99  # Discount factor for future rewards
LEARNING_RATE = 0.0005
ALPHA = 1.0  # Weight for throughput in reward function
BETA = 1.0   # Weight for RTT in reward function
GAMMA_R = 1.0  # Weight for jitter in reward function
DELTA = 0.1  # Weight for packet loss in reward function
ETA = 1.0    # Weight for bitrate in reward function

# Ideal values for testing
IDEAL_THROUGHPUT = 10  # Ideal throughput value (Mbps)
IDEAL_RTT = 20         # Ideal RTT value (ms)
IDEAL_BITRATE = 10     # Ideal bitrate value (Mbps)

# Define the Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.output_layer(x), dim=-1)
        return action_probs

# Define the Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.output_layer(x)
        return value

# SAC Agent
class SACAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        # Initialize the Actor and Critic Networks
        self.actor = ActorNetwork(STATE_SIZE, ACTION_SIZE)
        self.critic_1 = CriticNetwork(STATE_SIZE)
        self.critic_2 = CriticNetwork(STATE_SIZE)

        # Optimizers for both networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=LEARNING_RATE)

    def act_output(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        action_probs = self.actor(state_tensor).detach().numpy().flatten()
        return action_probs

    def learn(self, state, action_probs, reward, next_state):
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])

        # Calculate target value
        with torch.no_grad():
            next_value_1 = self.critic_1(next_state_tensor)
            next_value_2 = self.critic_2(next_state_tensor)
            next_value = torch.min(next_value_1, next_value_2)
            target_value = reward_tensor + GAMMA * next_value

        # Update Critic Networks (Value Loss)
        value_1 = self.critic_1(state_tensor)
        value_2 = self.critic_2(state_tensor)
        critic_loss_1 = F.mse_loss(value_1, target_value)
        critic_loss_2 = F.mse_loss(value_2, target_value)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_optimizer_2.step()

        # Update Actor Network (Policy Loss)
        action_probs_tensor = torch.FloatTensor(action_probs).unsqueeze(0)
        predicted_action_probs = self.actor(state_tensor)
        policy_loss = F.mse_loss(predicted_action_probs, action_probs_tensor)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return policy_loss.item(), critic_loss_1.item(), critic_loss_2.item()

    def batch_learn(self, items):
        # Convert batch to tensors
        states = torch.FloatTensor([item.state.to_state_vector() for item in items])
        action_probs = torch.FloatTensor([item.action_probs for item in items])
        rewards = torch.FloatTensor([item.reward for item in items])

        # Calculate target values and TD errors for the entire batch
        values_1 = self.critic_1(states[:-1])
        values_2 = self.critic_2(states[:-1])

        with torch.no_grad():
            # Ensure no in-place modification
            next_values_1 = self.critic_1(states[1:])
            next_values_2 = self.critic_2(states[1:])
            
            # Instead of doing in-place operations, perform safe out-of-place computations
            next_values = torch.min(next_values_1, next_values_2)
            target_values = rewards[1:].unsqueeze(1) + GAMMA * next_values

        # Update Critic Networks (Value Loss)
        critic_loss_1 = F.mse_loss(values_1, target_values)
        critic_loss_2 = F.mse_loss(values_2, target_values)

        # Ensure no in-place modification when updating gradients
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward(retain_graph=True)  # Retain graph to avoid in-place issues
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()  # No retain_graph needed for the second loss
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_optimizer_2.step()

        # Update Actor Network (Policy Loss)
        action_probs_tensor = action_probs
        predicted_action_probs = self.actor(states)

        # Make sure the loss calculation is not modifying the tensors in place
        policy_loss = F.mse_loss(predicted_action_probs, action_probs_tensor)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return policy_loss.item(), critic_loss_1.item(), critic_loss_2.item()


    def save_model(self, actor_path="actor_model.pth", critic_1_path="critic_1_model.pth", critic_2_path="critic_2_model.pth"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)

    def load_model(self, actor_path="actor_model.pth", critic_1_path="critic_1_model.pth", critic_2_path="critic_2_model.pth"):
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
        if os.path.exists(critic_1_path):
            self.critic_1.load_state_dict(torch.load(critic_1_path))
        if os.path.exists(critic_2_path):
            self.critic_2.load_state_dict(torch.load(critic_2_path))

# Reward function including RTT, jitter, packet loss, and bitrate
def calculate_reward(state: 'AggDelivState'):
    throughput_sum = 0
    rtt_sum = 0
    packet_loss_sum = 0
    bitrate_sum = state.bitrate  # Bitrate is stored at the state level
    jitter_sum = 0
    active_paths = 0

    for path in state.paths_info:
        if path.active:
            throughput_sum += path.historical_throughput
            rtt_sum += path.RTT
            packet_loss_sum += path.packet_loss
            jitter_sum += path.jitter
            active_paths += 1

    if active_paths == 0:
        return 0  # No active paths, no reward

    jitter_reward = 1 / (jitter_sum + 1)
    ideal_rtt_penalty = (rtt_sum / active_paths ) / IDEAL_RTT # Deviation from ideal RTT
    ideal_throughput_reward = (throughput_sum / active_paths ) / IDEAL_THROUGHPUT # Capped by ideal throughput
    packet_loss_reward = packet_loss_sum / active_paths  # Higher packet loss, lower reward
    ideal_bitrate_reward = bitrate_sum / IDEAL_BITRATE  # Capped by ideal bitrate

    reward = (ALPHA * ideal_throughput_reward) - (BETA * ideal_rtt_penalty) + (GAMMA_R * jitter_reward) - (DELTA * packet_loss_reward) + (ETA * ideal_bitrate_reward)
    return reward

# Class to represent an offline data item, encapsulating AggDelivState, action probabilities, and reward value
class TrainDataItem:
    def __init__(self, time, state, action_probs, reward):
        self.time = time
        self.state = state
        self.action_probs = action_probs
        self.reward = reward

# Class to process offline training data
class OfflineDataProcessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.historical_rtt_records = {i: deque(maxlen=20) for i in range(MAX_PATHS)}
        self.throughput_records = {i: deque(maxlen=20) for i in range(MAX_PATHS)}
        self.offline_data_items = []
        self.preprocess_data()

    def preprocess_data(self):
        # Calculate historical throughput and jitter during data loading
        temp_reward = None
        for index in reversed(range(len(self.data))):
            row = self.data.iloc[index]
            paths_info = []
            for i in range(MAX_PATHS):
                current_rtt = row[f"Path{i} RTT"]
                current_throughput = row[f"Path{i} throughput"]

                # Update historical RTT and calculate jitter
                self.historical_rtt_records[i].append(current_rtt)
                average_rtt = sum(self.historical_rtt_records[i]) / len(self.historical_rtt_records[i])
                jitter = abs(current_rtt - average_rtt)

                # Update historical throughput and calculate moving average
                self.throughput_records[i].append(current_throughput)
                historical_throughput = sum(self.throughput_records[i]) / len(self.throughput_records[i])

                path_info = PathInfo(
                    path_id=i,
                    active=row[f"Path{i} Active"],
                    RTT=current_rtt,
                    historical_throughput=historical_throughput,
                    packet_loss=row[f"Path{i} packet_loss_rate"],
                    bandwidth=row[f"Path{i} bandwidth"],
                    cwnd=row[f"Path{i} cwnd"],
                    jitter=jitter
                )
                paths_info.append(path_info)

            state = AggDelivState(resolution_type=row['Resolution Type'], paths_info=paths_info, bitrate=row['Bitrate'])
            action_probs = [row[f"Path{i} Frequency"] for i in range(MAX_PATHS)]
            reward = temp_reward if temp_reward is not None else 0
            temp_reward = calculate_reward(state)
            offline_data_item = TrainDataItem(time=row['Absolute Time'], state=state, action_probs=action_probs, reward=reward)
            self.offline_data_items.insert(0, offline_data_item)
            
        self.offline_data_items.reverse()

# Function to train the model using offline data
def train_offline(agent, data_processor, epochs=10, save_path="trained_model"):
    for epoch in range(epochs):
        state_vector = None
        action_probs = None
        reward = None
        next_state_vector = None
        for i in range(len(data_processor.offline_data_items)-1):
            item = data_processor.offline_data_items[i]
            if i == 0:
                state_vector = item.state.to_state_vector()
                action_probs = item.action_probs
                reward = item.reward
                continue
            next_state_vector = item.state.to_state_vector()
            actor_loss, critic_loss_1, critic_loss_2 = agent.learn(state_vector, action_probs, reward, next_state_vector)
            action_probs = item.action_probs
            reward = item.reward
            state_vector = next_state_vector
        print(f"Epoch {epoch+1}/{epochs}, Actor Loss: {actor_loss}, Critic Loss 1: {critic_loss_1}, Critic Loss 2: {critic_loss_2}")
        
    # Save the trained model after training
    agent.save_model()

def batch_train_offline(agent, data_processor, epochs=10, save_path="trained_model"):
    for epoch in range(epochs):
        length = len(data_processor.offline_data_items)
        for i in range(0, length, BATCH_SIZE-1):
            items = data_processor.offline_data_items[i:min(i+BATCH_SIZE, length)]
            actor_loss, critic_loss_1, critic_loss_2  = agent.batch_learn(items)
        print(f"Epoch {epoch+1}/{epochs}, Actor Loss: {actor_loss}, Critic Loss 1: {critic_loss_1}, Critic Loss 2: {critic_loss_2}")
        
    # Save the trained model after training
    agent.save_model()

# Load data and train the model
if __name__ == '__main__':
    filepath = "offline_data_realistic_50_modified.csv"
    data_processor = OfflineDataProcessor(filepath)
    agent = SACAgent()
    batch_train_offline(agent, data_processor, epochs=10)