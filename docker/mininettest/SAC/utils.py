MAX_PATHS = 4
BATCH_SIZE = 4
TIME_SLICE = 1
STATE_SIZE = 2 + MAX_PATHS * 7  # Resolution type + metrics for each path, excluding bitrate
ACTION_SIZE = MAX_PATHS  # Action represents the probability of assigning packets to each path
SAC_PORT = 12345
NETWORK_PORT = 12346