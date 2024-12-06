import time
import argparse
from basicTopo import setup_environment

SERVER_CMD = "cd /data/AggDeliv/mpquic-rl/MAppLE/dash/ && ./caddy -quic -mp -scheduler rl -fec --fecConfig win-xor"
CERTPATH = "--certpath /data/AggDeliv/mpquic-rl/docker/quic/quic_go_certs"
SCH = "-scheduler %s"
ARGS = "-bind :6121 -www /var/www/"
END = "> /data/AggDeliv/mpquic-rl/docker/logs/server.logs 2>&1"

BASIC_DELAY = 10

# CLIENT_CMD = "openssl s_client -connect 10.0.0.20:4242 -servername 10.0.0.20 > /data/AggDeliv/mpquic-rl/docker/logs/client.logs 2>&1"
# CLIENT_CMD = "python3 -m http.server --bind 10.0.0.20 --certfile /data/AggDeliv/mpquic-rl/MAppLE/dash/cert.pem --keyfile /data/AggDeliv/mpquic-rl/MAppLE/dash/privkey.pem 4242 && curl -k https://10.0.0.20:4242/ > /data/AggDeliv/mpquic-rl/docker/logs/client.logs 2>&1"
CLIENT_CMD = "cd /data/AggDeliv/mpquic-rl/docker/logs && python3 /data/AggDeliv/mpquic-rl/MAppLE/astream/dash_client.py -q -mp -m https://10.0.0.20:4242/manifest.mpd -s rtt -p basic > /data/AggDeliv/mpquic-rl/docker/logs/client.logs 2>&1"
# CLIENT_CMD = "cd /data/AggDeliv/mpquic-rl/docker/logs && python3 /data/AggDeliv/mpquic-rl/MAppLE/astream/dash_client.py -q -mp -m http://10.0.0.20:4040/manifest.mpd -s rtt > /data/AggDeliv/mpquic-rl/docker/logs/client.logs 2>&1"

# CLIENT_CMD = "cd /data/AggDeliv/mpquic-rl/MAppLE/astream/ && python dash_client.py -m http://10.0.0.20:4040/manifest.mpd -p basic > /data/AggDeliv/mpquic-rl/docker/logs/client.logs 2>&1"

TCP_SERVER_CMD = "cd /data/AggDeliv/mpquic-rl/docker/var/www && python -m SimpleHTTPServer 80 &"
TCP_CLIENT_CMD = "curl -s -o /dev/null 10.0.0.20/demo &"


def setup():
    net = setup_environment()
    net.start()
    return net


def exec_test(server_cmd, rtt, tcp_traffic):
    network = setup()

    s1 = network.get("s1")
    server = network.get("server")
    client = network.get("client")

    if tcp_traffic:
        server.cmd(TCP_SERVER_CMD)

    server.sendCmd(server_cmd)

    time.sleep(2)
    s1.cmd("./scripts/set_delay.bash %d" % int((BASIC_DELAY + rtt) / 2))
    client.cmd("./scripts/client_set_delay.bash %d" % int((BASIC_DELAY + rtt) / 2))

    if tcp_traffic:
        client.cmd(TCP_CLIENT_CMD)

    start = time.time()
    server_pcap = server.popen('tcpdump -w /data/AggDeliv/mpquic-rl/MappLE/docker/logs/h1_dump.pcap')
    client.sendCmd(CLIENT_CMD)
    # Timeout of 20 seconds for detecting crashing tests
    print("Wait for Client Output")
    output = client.monitor(timeoutms=20000)
    # Check for timeout
    if client.waiting:
        print("Client Timeout")
        delta = 20
        client.sendInt()
        client.waiting = False
        server_pcap.terminate()
 
        network.stop()
        time.sleep(1)
        network.cleanup()
    else:
        print("Client end")
        # TODO: Check for errors here?? How??
        delta = time.time() - start

    server.sendInt()

    server.monitor()
    server.waiting = False


def do_training(sch, rtt, tcp_b):
    server_cmd = " ".join([SERVER_CMD, END])

    exec_test(server_cmd, rtt, tcp_b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executes a test with defined scheduler')
    parser.add_argument('--scheduler', dest="sch", help="Scheduler (rtt, random)", required=True)
    parser.add_argument('--rtt', type=int, dest="rtt", help="rtt primary leg")
    parser.add_argument('--background-tcp', dest="tcp_background", action="store_true",
                        help='generates TCP background traffic during tests')

    args = parser.parse_args()
    do_training(args.sch, args.rtt, args.tcp_background)
