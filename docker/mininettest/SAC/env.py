from utils import MAX_PATHS

class PathInfo:
    def __init__(self, path_id, active, RTT, historical_throughput, packet_loss, bandwidth, cwnd, jitter):
        self.path_id = path_id
        self.active = active
        self.RTT = RTT
        self.historical_throughput = historical_throughput
        self.packet_loss = packet_loss
        self.bandwidth = bandwidth
        self.cwnd = cwnd
        self.jitter = jitter
        
    def to_dict(self):
        return {
            "path_id": self.path_id,
            "active": self.active,
            "RTT": self.RTT,
            "historical_throughput": self.historical_throughput,
            "packet_loss": self.packet_loss,
            "bandwidth": self.bandwidth,
            "cwnd": self.cwnd,
            "jitter": self.jitter
        }
        
    @staticmethod
    def from_dict(data):
        return PathInfo(
            path_id=data["path_id"],
            active=data["active"],
            RTT=data["RTT"],
            historical_throughput=data["historical_throughput"],
            packet_loss=data["packet_loss"],
            bandwidth=data["bandwidth"],
            cwnd=data["cwnd"],
            jitter=data["jitter"]
        )

# AggDelivState class
class AggDelivState:
    def __init__(self, resolution_type, paths_info, bitrate):
        self.resolution_type = resolution_type
        self.paths_info = paths_info
        self.bitrate = bitrate
    
    def to_state_vector(self):
        state_vector = [self.resolution_type, self.bitrate]
        for path in self.paths_info:
            state_vector.extend([
                path.active,
                path.RTT,
                path.historical_throughput,
                path.packet_loss,
                path.bandwidth,
                path.cwnd,
                path.jitter
            ])
        return state_vector
    
    def to_dict(self):
        return {
            "resolution_type": self.resolution_type,
            "bitrate": self.bitrate,
            "paths_info": [path.to_dict() for path in self.paths_info]  # Convert each PathInfo to a dict
        }
        
    @staticmethod
    def from_dict(data):
        paths_info = [PathInfo.from_dict(path) for path in data["paths_info"]]
        return AggDelivState(
            resolution_type=data["resolution_type"],
            paths_info=paths_info,
            bitrate=data["bitrate"]
        )
