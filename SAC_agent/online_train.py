import socket
import json
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import RemoteController
from mininet.link import TCLink
from scheduler import PacketScheduler
from monitor import collect_network_state
import time
import threading
from utils import NETWORK_PORT

class MultiPathTopo(Topo):
    def build(self):
        h1 = self.addHost('h1')  # Mobile phone
        h2 = self.addHost('h2')  # Remote server

        s1 = self.addSwitch('s1')  # WiFi path
        s2 = self.addSwitch('s2')  # 4G path
        s3 = self.addSwitch('s3')  # LTE path
        s4 = self.addSwitch('s4')  # Optional additional path

        self.addLink(h1, s1, bw=1000, delay='5ms', loss=0, intfName1='h1-s1')
        self.addLink(h1, s2, bw=500, delay='30ms', loss=2, intfName1='h1-s2')
        self.addLink(h1, s3, bw=100, delay='100ms', loss=5, intfName1='h1-s3')
        self.addLink(h1, s4, bw=70, delay='150ms', loss=5, intfName1='h1-s4')

        self.addLink(h2, s1, bw=1000, delay='5ms', loss=0, intfName1='h2-s1')
        self.addLink(h2, s2, bw=500, delay='30ms', loss=2, intfName1='h2-s2')
        self.addLink(h2, s3, bw=100, delay='100ms', loss=5, intfName1='h2-s3')
        self.addLink(h2, s4, bw=70, delay='150ms', loss=5, intfName1='h2-s4')

# def run():
#     topo = MultiPathTopo()
#     controller = RemoteController('c0', ip='127.0.0.1', port=6653)
#     net = Mininet(topo=topo, controller=controller, link=TCLink)

#     # Start network
#     net.start()
#     h1 = net.get('h1')
#     h2 = net.get('h2')
#     scheduler = PacketScheduler()

#     # Default action probabilities
#     action_prob = [0.45, 0.3, 0.15, 0.1]
#     scheduler.update_action_prob(action_prob)
    
#     total_packets = 1000
#     start_time = time.time()

#     # Socket Client to communicate with SAC server
#     sac_host = 'localhost'
#     sac_port = 12345

#     for i in range(total_packets):
#         # Every few packets, collect network state and send to SAC
#         if request_received == true:
#             state = collect_network_state(net) 
#             state_vector = state.to_state_vector()

#             # Communicate with SAC server
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 s.connect((sac_host, sac_port))
#                 s.send(json.dumps({"state_vector": state_vector}).encode('utf-8'))
#                 data = s.recv(1024)
#                 action_probs = json.loads(data.decode('utf-8'))["action_probs"]

#             # Update scheduler with new action probabilities
#             scheduler.update_action_prob(action_probs)

#         # Distribute the packet according to the scheduler's action probabilities
#         path = scheduler.distribute_packet()
#         if path == 0:
#             print("Sending packet from h1 to h2 via Path 0 (WiFi)")
#             h1.cmd(f"ping -c 1 {h2.IP()}")
#         elif path == 1:
#             print("Sending packet from h1 to h2 via Path 1 (4G)")
#             h1.cmd(f"ping -c 1 {h2.IP()}")
#         elif path == 2:
#             print("Sending packet from h1 to h2 via Path 2 (LTE)")
#             h1.cmd(f"ping -c 1 {h2.IP()}")
#         elif path == 3:
#             print("Sending packet from h1 to h2 via Path 3 (LTE)")
#             h1.cmd(f"ping -c 1 {h2.IP()}")

#     net.stop()

class NetworkEnvironment:
    def __init__(self, host='localhost', port=NETWORK_PORT):
        self.host = host
        self.port = port
        self.action_probs = [0.25, 0.25, 0.25, 0.25]  # Default probabilities
        self.net = Mininet(topo=MultiPathTopo(), controller=None, link=TCLink)

    def start_network(self):
        """Start the Mininet network."""
        self.net.start()
        print("Mininet network started.")

    def stop_network(self):
        """Stop the Mininet network."""
        if self.net:
            self.net.stop()
            print("Mininet network stopped.")

    def handle_client(self, conn):
        """Handle requests from the SACServer."""
        data = conn.recv(4096)
        if not data:
            return
        message = json.loads(data.decode('utf-8'))

        if message.get("request") == "network_state":
            state = collect_network_state(self.net)
            conn.send(json.dumps(state.to_dict()).encode('utf-8'))
        elif "action_probs" in message:
            self.action_probs = message["action_probs"]
            print(f"Updated action probabilities: {self.action_probs}")

        conn.close()

    def start_server(self):
        """Start the server to communicate with the SACServer."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Network environment server started at {self.host}:{self.port}")

        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=self.handle_client, args=(conn,)).start()

if __name__ == "__main__":
    env = NetworkEnvironment()
    try:
        env.start_network()
        env.start_server()
    except KeyboardInterrupt:
        env.stop_network()
