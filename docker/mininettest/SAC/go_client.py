import grpc
import time
import random
import mpquic_pb2
import mpquic_pb2_grpc

def get_network_state(stub):
    response = stub.GetNetworkState(mpquic_pb2.Empty())
    print("Received Network State:")
    for path in response.paths:
        print(f"Path {path.path_id} -> RTT: {path.RTT}, CWND: {path.cwnd}")
    return response

def send_policy(stub, probabilities):
    policy = mpquic_pb2.Policy(path_probabilities=probabilities)
    stub.SendPolicy(policy)
    print(f"Sent policy: {probabilities}")

def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = mpquic_pb2_grpc.MPQUICServiceStub(channel)

    while True:
        # Pull the current network state
        state = get_network_state(stub)

        # Compute a policy (example: random probabilities summing to 1)
        num_paths = len(state.paths)
        probabilities = [random.random() for _ in range(num_paths)]
        total = sum(probabilities)
        normalized_probabilities = [p / total for p in probabilities]

        # Send the updated policy to the scheduler
        send_policy(stub, normalized_probabilities)

        time.sleep(1)  # Wait for the next update cycle

if __name__ == "__main__":
    main()
