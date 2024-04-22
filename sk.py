import numpy as np
from communication import Logger, Average
from typing import Union
import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import json
import jsonpickle

from helpers import Q

class GaussianChannel:
    def __init__(self, sigma) -> None:
        """Defines a container for BSC with error probability p"""
        self.sigma = sigma
    
    def rx(self, tx: np.float64):
        return tx + np.random.normal(0, self.sigma, 1).item()


class SKscheme:
    def __init__(self, channel: GaussianChannel, block_size, N, W = None, alpha = 'optim', tx_bitstream = np.array([]), rx_bitstream = np.array([])) -> None:
        '''Simulate the performance of SK scheme'''
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


class SK_Simple(Logger): # Notes notation
    def __init__(self,
                 snr = 4, N = 20, 
                 block_size = 8,
                 enable: bool = False, 
                 tx_bitstream = np.array([]), 
                 rx_bitstream = np.array([])) -> None:
        '''Simulate the performance of SK scheme'''
        super().__init__(enable=enable)
        self.X: np.ndarray = np.array([])
        self.Y: np.ndarray = np.array([])
        self.n = 1
        self.N = N  # Transmissions ber block
        self.block_size = block_size
        self.snr = snr

        self.rate = block_size/N

        self.theta: np.float64 = 0

        self.tx_bitstream: np.ndarray = tx_bitstream
        self.tx_bitstream_idx = 0
        self.rx_bitstream: np.ndarray = rx_bitstream

        self.num_blocks = self.tx_bitstream.size // self.block_size
        # assert not (self.tx_bitstream.size % self.block_size), "The bitstream size should be divisible by block size"

        self.powers2 = 2**np.arange(self.block_size)
        self.M = 2**self.block_size

        self.num_error = 0

    def get_next_word(self) -> Union[np.ndarray, None]:
        if self.tx_bitstream_idx + self.block_size <= self.tx_bitstream.size:
            ret = self.tx_bitstream[self.tx_bitstream_idx:self.tx_bitstream_idx + self.block_size]
            self.tx_bitstream_idx+=self.block_size
            return ret
        return None
    
    def theta_to_seq(self, theta):
        """Convert a point on a message line into a bit array"""
        res = []
        temp = theta
        for _ in range(self.block_size):
            res.append(1 if temp >= 0.5 else 0)
            temp*=2
            if temp >= 1:
                temp -= 1
        return np.array(res)
    

    def sim_block(self):
        # Determine if the symbol was seccesfully recieved by the reciever and
        # Either proceed to the next one, or keep transmitting the bits for previous symbol

        snr = self.snr
        N = self.N

        block = self.get_next_word()
        if block is None:
            return "done"
        n = block.size
        number = np.dot(np.flip(block), self.powers2[:n])
        theta = ((number + 1/2) / self.M)
        
        X = np.zeros(N)
        Z = np.random.normal(0, 1, N)
        X[0] = theta
        summand = np.power(np.sqrt(1 + snr), np.arange(N-1)) * Z[1:]
        temp_sum = np.concatenate(([0], np.cumsum(summand)[:-1]))
        X[1:] = - np.sqrt(snr / np.power(1 + snr, np.arange(N-1))) * (Z[0] + np.sqrt(snr) * temp_sum)  
        Y = X + Z
        summand = np.power(1 + snr, -np.arange(2, N+1)/2) * Y[1:]
        temp_sum = sum(summand)
        theta_est =  Y[0] + np.sqrt(snr) * temp_sum

        # Get bits back
        detect = self.theta_to_seq(theta_est)
        if (detect != block).any():
            self.num_error += 1

        self.log(theta, theta_est, type(theta))

        self.rx_bitstream = np.append(self.rx_bitstream, detect)

def test_N_sweep(N_sweep = np.linspace(2, 10, 9).astype(np.int32)):
    '''Run Horstein scheeme for various probabilities of error'''
    r = Parallel(n_jobs=20)(delayed(lambda x, i: simulate_sk(N = x, pos = i))(N, i) for i, N in enumerate(N_sweep))
    return r

def simulate_sk(snr = 11, l = 100, rate = 1, N = 5, pos = 0, enable = False):
    capacity = 1/2 * np.log2(1 + snr)
    print(f"Rate: {rate}, capacity {capacity}")
    if capacity < rate:
        raise ValueError("Capacity should be bigger than rate")
    sk = SK_Simple(tx_bitstream=np.random.randint(0, 2, l), snr = snr, N = N, block_size=int(N*rate), enable = enable)
    
    i = 0
    # for i in range(100):
    # progress_bar = tqdm.tqdm(total=l, desc="Progress", leave=True, position=pos)
    while sk.rx_bitstream.size < l:
        i += 1
        # progress_bar.n = sk.rx_bitstream.size
        # progress_bar.refresh()
        # progress_bar.set_description("Iteration {}".format(i))
        if sk.sim_block() == "done":
            break
        

    # progress_bar.close()
    errors = np.sum(np.abs(sk.tx_bitstream[:sk.rx_bitstream.size] - sk.rx_bitstream))
    # print("Decoded wrong: ", np.sum(np.abs(horst.tx_bitstream[:l] - horst.rx_bitstream[:l])))
    return {
        "sent": sk.tx_bitstream[:l],
        "recieved": sk.tx_bitstream[:l],
        "error": errors / l,
        "N": N,
        "Pe": sk.num_error / sk.num_blocks,
        "Pe_mod": sk.num_error / l,
    }


def plot_theoretical(snr = 10, N = np.linspace(2, 10, 9), R = 1):
    capacity = 1/2 * np.log2(1 + snr)
    assert capacity > R
    Pe = 2 * (2**(N*R) - 1) / 2**(N*R) * Q(np.sqrt(6 * (1 + snr) ** (N-1) / (2 ** (2 * N * R) - 1)))
    print(Pe)

    plt.plot(N, -np.log(Pe))
    plt.savefig(f"sk_theoretical.png")



if __name__ == "__main__":
    """Perform testin of some of the funcitons"""
    np.set_printoptions(precision=25)
    print(simulate_sk())
    # plot_theoretical()
    # r = test_N_sweep()
    # print(r)
    # serialized_data = jsonpickle.encode(r, unpicklable=False)

    # file_path = 'r_sk.json'
    # with open(file_path, 'w') as outfile:
    #     outfile.write(serialized_data)

    # # with open(file_path, 'r') as infile:
    # #     json_data = infile.read()

    # # # Deserialize using jsonpickle
    # # data = jsonpickle.decode(json_data)

    # fig, ax1 = plt.subplots()

    # ax1.plot([a['N'] for a in r], -np.log([a['error'] for a in r]))
    

    # # plt.xscale('log')
    # ax1.set_xlabel("N")
    # ax1.set_ylabel("-log10(Pe)")
    # ax1.grid(True)
    # fig.savefig(f"sk_sim.png")


