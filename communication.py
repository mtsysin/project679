"""Contains all of the code needed to simulate the channel in general"""
import numpy as np

class System:
    def __init__(self, channel, tx_size = 1024) -> None:
        self.tx_size = tx_size
        self.transmit: np.ndarray = np.random.randint(0, 2, tx_size)
        self.channel = channel

    def simulate(self):
        pass
        

class BSC:
    def __init__(self, p) -> None:
        """Defines a container for BSC with error probability p"""
        self.p = p
    
    def rx(self, tx: bool):
        return (not tx) if np.random.rand() < self.p else tx

class Horstein:
    def __init__(self, p,) -> None:
        pass