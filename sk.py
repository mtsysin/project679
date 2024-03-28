import numpy as np


class GaussianChannel:
    def __init__(self, sigma) -> None:
        """Defines a container for BSC with error probability p"""
        self.sigma = sigma
    
    def rx(self, tx: np.float64):
        return tx + np.random.normal(0, self.sigma, 1).item()


class SKscheme:
    def __init__(self, channel: GaussianChannel, block_size, N, W = None, alpha = 'optim', tx_bitstream = np.array([]), rx_bitstream = np.array([])) -> None:
        self.channel = channel
        self.block_size = block_size
        self.alpha = alpha if alpha != 'optim' else self.optim_alpha()
        self.X: np.ndarray = np.array([0.5])
        self.Y: np.ndarray = np.array([])
        self.n = 1
        self.N = N  # Limit of transmisions per symbol
        self.tx_val: np.float64 # placeholder for transmitted value

        self.W = W # bandwidth, None is unlimited

        self.current_sequence = np.array([])
        self.theta: np.float64 = 0

        self.tx_bitstream: np.ndarray = tx_bitstream
        self.tx_bitstream_idx = 0
        self.rx_bitstream: np.ndarray = rx_bitstream

    def optim_alpha(self):
        N0 = 2 * self.channel.sigma**2

    def get_next_word(self):
        if self.tx_bitstream_idx + self.block_size <= self.tx_bitstream.size:
            self.current_sequence = self.tx_bitstream[self.tx_bitstream_idx:self.tx_bitstream_idx + self.block_size]
            self.tx_bitstream_idx+=self.block_size
            return True
        return False
    
    def theta_to_seq(self):
        """Convert a point on a message line into a bit array"""
        res = []
        temp = self.theta
        for _ in range(self.block_size):
            res.append(1 if temp >= 0.5 else 0)
            temp*=2
            if temp >= 1:
                temp -= 1
        return np.array(res)


    def tx(self):
        # Determine if the symbol was seccesfully recieved by the reciever and
        # Either proceed to the next one, or keep transmitting the bits for previous symbol

        if self.n == self.N + 1:
            self.n = 1
            self.get_next_word()
            self.X = np.array([0.5])
            self.Y = np.array([])

            # Find the message point
            assert self.block_size == self.current_sequence.shape[0]
            M = 2**self.block_size # number of possible sequences in current transmission
        
            number = np.dot(np.flip(self.current_sequence), 2**np.arange(self.block_size))
            self.theta = (number + 1/2) / M

        self.tx_val = self.alpha(self.X[-1] - self.theta)

    def rx(self):
        self.Y = np.append(self.Y, self.channel(self.tx_val))
        self.X = np.append(self.X, self.X[-1] - (1/(self.alpha * self.n))*self.Y[-1])
        if self.n == self.N + 1:
            # Record current bit
            self.rx_bitstream = np.append(self.rx_bitstream, self.theta_to_seq())
        self.n += 1


if __name__ == "__main__":
    """Perform testin of some of the funcitons"""
    np.set_printoptions(precision=25)
    LEN = 4096
    sk = SKscheme(channel=GaussianChannel(0.1), tx_bitstream=np.random.randint(0, 2, LEN))

    i = 0
    # for i in range(100):
    while sk.rx_bitstream.size < LEN:
        i += 1
        print(i, "###############################")
        sk.tx()
        sk.rx()


    print(f"Were sending: {sk.tx_bitstream[:LEN]}")
    print(f"Recieved: {sk.rx_bitstream[:LEN]}")

    print(np.sum(np.abs(sk.tx_bitstream[:LEN] - sk.rx_bitstream[:LEN])))