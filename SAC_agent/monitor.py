from env import PathInfo, AggDelivState
import re, threading

host1_lock = threading.Lock()

def safe_cmd(host, command):
    with host1_lock:
        return host.cmd(command)

def get_rtt(host1, host2, link_name):
    """
    Get RTT between host1 and host2 over a specific link.
    """
    try:
        print(f"Measuring RTT between {host1.name} and {host2.name} over {link_name}")
        # Optionally set routing to ensure traffic uses the specific path
        safe_cmd(host1, f"ip route del {host2.IP()} dev {link_name}")
        result = safe_cmd(host1, f"ping -c 2 {host2.IP()}")
        match = re.search(r'rtt min/avg/max/mdev = [^/]+/([^/]+)/', result)
        rtt = float(match.group(1)) if match else 0.0
        return rtt
    except Exception as e:
        print(f"Error measuring RTT: {e}")
        return 0.0
    finally:
        # Clean up the route
        safe_cmd(host1, f"ip route del {host2.IP()} dev {link_name}")



def get_throughput(host1, host2, link_name):
    """
    Get throughput between host1 and host2 over a specific link.
    """
    try:
        print(f"Measuring throughput between {host1.name} and {host2.name} over {link_name}")
        safe_cmd(host1, f"ip route add {host2.IP()} dev {link_name}")
        safe_cmd(host2, "iperf -s &")
        result = safe_cmd(host1, f"iperf -c {host2.IP()} -t 2")
        match = re.search(r'(\d+\.?\d*) Mbits/sec', result)
        throughput = float(match.group(1)) if match else 0.0
        return throughput
    except Exception as e:
        print(f"Error measuring throughput: {e}")
        return 0.0
    finally:
        safe_cmd(host1, f"ip route del {host2.IP()} dev {link_name}")
        safe_cmd(host2, "pkill -f iperf")



def get_packet_loss(host1, host2, link_name):
    """
    Get packet loss percentage between host1 and host2 over a specific link.
    """
    try:
        print(f"Measuring packet loss between {host1.name} and {host2.name} over {link_name}")
        safe_cmd(host1, f"ip route add {host2.IP()} dev {link_name}")
        result = safe_cmd(host1, f"ping -c 5 {host2.IP()}")
        match = re.search(r'(\d+)% packet loss', result)
        loss = float(match.group(1)) if match else 0.0
        return loss
    except Exception as e:
        print(f"Error measuring packet loss: {e}")
        return 0.0
    finally:
        safe_cmd(host1, f"ip route del {host2.IP()} dev {link_name}")



def get_cwnd(host1, link_name):
    """
    Get the congestion window size for the given host via the specified switch.
    """
    print(f"Getting congestion window for {host1.name} via {link_name}")
    cwnd = 10  # Placeholder value
    return cwnd


def parallel_metric_collection(host1, host2, link_name, results, index):
    rtt = get_rtt(host1, host2, link_name)
    throughput = get_throughput(host1, host2, link_name)
    packet_loss = get_packet_loss(host1, host2, link_name)
    cwnd = get_cwnd(host1, link_name)
    results[index] = (rtt, throughput, packet_loss, cwnd)

def collect_network_state(net):
    paths_info = []
    h1 = net.get('h1')
    h2 = net.get('h2')

    # Define paths
    links = ['h1-s1', 'h1-s2', 'h1-s3', 'h1-s4']
    bandwidths = [1000, 500, 100, 70]
    frequencies = [0.1, 0.2, 0.3, 0.4]

    # Shared list to store results
    results = [None] * len(links)
    threads = []

    # Create a thread for each path
    for i, link in enumerate(links):
        t = threading.Thread(target=parallel_metric_collection, args=(h1, h2, link, results, i))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Process results
    for i, link in enumerate(links):
        rtt, throughput, packet_loss, cwnd = results[i]
        paths_info.append(PathInfo(
            path_id=i,
            active=1,
            RTT=rtt,
            historical_throughput=throughput,
            packet_loss=packet_loss,
            bandwidth=bandwidths[i],
            cwnd=cwnd,
            jitter=5
        ))

    resolution_type = 0
    bitrate = sum(path.historical_throughput for path in paths_info)
    state = AggDelivState(
        resolution_type=resolution_type,
        paths_info=paths_info,
        bitrate=bitrate
    )
    return state

