import socket
import json
from SAC import SACAgent, calculate_reward, TrainDataItem
from utils import MAX_PATHS, STATE_SIZE, ACTION_SIZE, TIME_SLICE, BATCH_SIZE, SAC_PORT, NETWORK_PORT
import time
import random
import pandas as pd
import numpy as np
from env import AggDelivState


class OnlineDataProcessor:
    def __init__(self):
        self.online_data_items = []  # Store collected online data

    def add_data(self, online_data_item):
        self.online_data_items.append(online_data_item)
        self.to_file()

    def get_data(self):
        if len(self.online_data_items) < BATCH_SIZE:
            return None
        else:
            # Randomly select a batch of data
            chosen_list = random.sample(self.online_data_items, BATCH_SIZE)
            return chosen_list
    
    def to_file(self, filename="online_data.csv"):
        """
        Save the collected online data into a CSV file in the same format as the offline data.
        """
        print("online data store")
        data = {
            "Absolute Time": [],
            "Resolution Type": [],
            "Bitrate": []
        }

        # For each path, we'll add corresponding information
        for i in range(MAX_PATHS):
            data[f"Path{i} Active"] = []
            data[f"Path{i} RTT"] = []
            data[f"Path{i} throughput"] = []
            data[f"Path{i} packet_loss_rate"] = []
            data[f"Path{i} bandwidth"] = []
            data[f"Path{i} cwnd"] = []
            data[f"Path{i} jitter"] = []
            data[f"Path{i} Frequency"] = []

        # Collect data from the online_data_items
        for item in self.online_data_items:
            data["Absolute Time"].append(item.time)
            data["Resolution Type"].append(item.state.resolution_type)
            data["Bitrate"].append(item.state.bitrate)
            
            for i in range(MAX_PATHS):
                path_info = item.state.paths_info[i] if i < len(item.state.paths_info) else None
                # For paths that exist, populate the data fields
                if path_info:
                    data[f"Path{i} Active"].append(path_info.active)
                    data[f"Path{i} RTT"].append(path_info.RTT)
                    data[f"Path{i} throughput"].append(path_info.historical_throughput)
                    data[f"Path{i} packet_loss_rate"].append(path_info.packet_loss)
                    data[f"Path{i} bandwidth"].append(path_info.bandwidth)
                    data[f"Path{i} cwnd"].append(path_info.cwnd)
                    data[f"Path{i} jitter"].append(path_info.jitter)
                else:
                    # In case the path doesn't exist, use default values
                    data[f"Path{i} Active"].append(0)
                    data[f"Path{i} RTT"].append(np.nan)
                    data[f"Path{i} throughput"].append(np.nan)
                    data[f"Path{i} packet_loss_rate"].append(np.nan)
                    data[f"Path{i} bandwidth"].append(np.nan)
                    data[f"Path{i} cwnd"].append(np.nan)
                    data[f"Path{i} jitter"].append(np.nan)
            
            # Action probabilities (assuming action_probs are provided)
            data["Path0 Frequency"].append(item.action_probs[0] if len(item.action_probs) > 0 else 0)
            data["Path1 Frequency"].append(item.action_probs[1] if len(item.action_probs) > 1 else 0)
            data["Path2 Frequency"].append(item.action_probs[2] if len(item.action_probs) > 2 else 0)
            data["Path3 Frequency"].append(item.action_probs[3] if len(item.action_probs) > 3 else 0)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Online data saved as {filename}")


class SACServer:
    def __init__(self, host='localhost', port=SAC_PORT, env_host='localhost', env_port=NETWORK_PORT):
        self.host = host
        self.port = port
        self.env_host = env_host
        self.env_port = env_port
        self.sac_agent = SACAgent()
        self.data_processor = OnlineDataProcessor()
        self.step = 0
    
    def update(self, time, state):
        print("update")
        action_probs = self.sac_agent.act_output(state.to_state_vector())  # Get action probabilities from the RL agent
        
        # Calculate reward from state
        print("reward")
        reward = calculate_reward(state)  # Reward based on state
        
        # Create a new data item and add it to the processor
        print("data item create")
        new_data_item = TrainDataItem(time, state, action_probs, reward)
        self.data_processor.add_data(new_data_item)
        
        # If enough data has been collected, learn from the batch
        update_list = self.data_processor.get_data()
        if update_list is not None:
            actor_loss, critic_loss_1, critic_loss_2 = self.sac_agent.batch_learn(update_list)  # Train the agent with the new data batch
            
        self.step += 1
        if self.step % BATCH_SIZE == 0:
            print(f"Step {self.step}, Actor Loss: {actor_loss}, Critic Loss 1: {critic_loss_1}, Critic Loss 2: {critic_loss_2}")
        
        return action_probs
    
    def fetch_network_state(self):
        """Request network state from the simulation environment."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.env_host, self.env_port))
            s.send(json.dumps({"request": "network_state"}).encode('utf-8'))
            print("get response")
            response = s.recv(4096)  # Increase buffer size if needed
            print("got response")
            return json.loads(response.decode('utf-8'))

    def send_action_probabilities(self, action_probs):
        """Send action probabilities to the network simulation."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.env_host, self.env_port))
            s.send(json.dumps({"action_probs": action_probs}).encode('utf-8'))

    def start_server(self):
        """Run the SAC server to interact with the network environment."""
        print(f"SAC Server started at {self.host}:{self.port}")
        start_time = time.time()
        while True:
            time.sleep(TIME_SLICE)
            try:
                # Fetch the current network state
                state_dict = self.fetch_network_state()
                
                state = AggDelivState.from_dict(state_dict)

                # Calculate action probabilities using SAC
                relative_time = time.time() - start_time
                action_probs = self.update(relative_time, state)

                # Send updated action probabilities to the network simulation
                self.send_action_probabilities(action_probs.tolist())
                
                print(f"Updated action probabilities sent: {action_probs}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    server = SACServer()
    server.start_server()