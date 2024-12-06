import random
from utils import MAX_PATHS 

class PacketScheduler:
    def __init__(self):
        self.action_size = MAX_PATHS  # Number of paths
        self.action_prob = [0] * MAX_PATHS  # Placeholder for action probabilities
    
    def update_action_prob(self, new_action_prob):
        """
        Update the action probabilities based on the RL agent's output.
        """
        self.action_prob = new_action_prob
    
    def distribute_packet(self):
        """
        Distribute each packet to one of the paths based on the action probabilities.
        For each packet, we randomly select a path based on the action_prob.
        """
        
        # Randomly choose a path based on the probabilities
        path = random.choices(range(self.action_size), weights=self.action_prob, k=1)[0]
        return path