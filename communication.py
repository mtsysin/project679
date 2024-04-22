"""Contains all of the code needed to simulate the channel in general"""
import numpy as np
import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import json
import jsonpickle

from helpers import*

class System:
    def __init__(self, channel, tx_size = 1024) -> None:
        self.tx_size = tx_size
        self.transmit: np.ndarray = np.random.randint(0, 2, tx_size)
        self.channel = channel

    def simulate(self):
        pass
        

class Logger:
    def __init__(self, enable: bool) -> None:
        self.enable = enable

    def log(self, *args, **kwargs):
        if self.enable:
            print(*args, **kwargs)

class Average:
    def __init__(self) -> None:
        self.num = 0
        self.avg = 0

    def update(self, val):
        s = self.num * self.avg + val
        self.num += 1
        self.avg = s / self.num

    def __repr__(self):
        return self.avg
    
    def __str__(self):
        return self.avg
    

class BSC:
    def __init__(self, p) -> None:
        """Defines a container for BSC with error probability p"""
        self.p = p
        self.q = 1-p
    
    def rx(self, tx: bool):
        return (not tx) if np.random.rand() < self.p else tx
    
    def capacity(self):
        return 1 - Hb(self.p)
    
class OrthogonalCoding(Logger):
    def __init__(self, enable: bool, rate: int) -> None:
        """This class is a crude simulation of very simple orthonogal coding scheme"""
        super().__init__(enable)



class Horstein(Logger):
    def __init__(self, tx_bitstream = np.array([]), rx_bitstream = np.array([]), channel: BSC = None, Pe = 0.01, d0 = 0.00001, enable = False) -> None:
        super().__init__(enable=enable)
        self.channel = channel
        # initialize the interval
        # midpoint positions is an array like:
        # Actual1, Actual2, ...
        # pos in stretched 1, pos in stretched 2, ...
        self.midpoint_positions: np.ndarray = np.array([[0.0, 0.0], [1.0, 1.0]]) # array of: actual_value, position after stretching
        self.Pe = Pe
        self.current_sequence = np.array([])
        self.time_in_queue = np.array([])
        self.tx_bitstream: np.ndarray = tx_bitstream
        self.rx_bitstream: np.ndarray = rx_bitstream
        self.rx_bit = None
        self.tx_bit = None
        self.d0 = d0
        self.tx_bitstream_idx = 0
        self.rx_decoded = None

        # Evaluation metrics
        self.N_average = Average()

    def get_additional_bit(self):
        if self.tx_bitstream_idx < self.tx_bitstream.size:
            extra_bit = self.tx_bitstream[self.tx_bitstream_idx]
            self.tx_bitstream_idx+=1
            self.current_sequence = np.append(self.current_sequence, extra_bit)
            self.time_in_queue = np.append(self.time_in_queue, 0)
            return True
        return False

    def find_in_midpoints(self, mode = "start_actual", split_point = 0.5):
        '''
        Function that finds appropriate positions of points either in stretcehd or actual sequences
        mode: starrt actual if we want to search through the actual values.
                start_stretch if we want to search throght the stretched interval.
        '''
        if mode == "start_stretch":
            row, row_second = 1, 0
        elif mode == "start_actual":
            row, row_second = 0, 1

        midpoints_base = self.midpoint_positions[:, row] # positions in the starting sequecne
        sorted_midpoints_base = np.sort(midpoints_base)
        # Find the index of an interval in the sorted array where the desirde point is:
        idx_split = np.searchsorted(sorted_midpoints_base, split_point, side='left', sorter=None)
        a_base, b_base = sorted_midpoints_base[idx_split-1], sorted_midpoints_base[idx_split] # get the interval
        a_ind, b_ind = np.where(midpoints_base == a_base), np.where(midpoints_base == b_base) # get the indices of these values in the base sequence
        try:
            # Figure out the corresponding interval on the other side:
            a_second, b_second = self.midpoint_positions[a_ind[0].item(), row_second], self.midpoint_positions[b_ind[0].item(), row_second]
        except:
            raise ValueError(f"There are none or multiple values for a_base, b_base: {self.midpoint_positions}, {a_base, b_base}")

        return (a_base, b_base), (a_ind, b_ind), (a_second, b_second)


    def update_reciever_dist(self):
        '''This function performs an update on the reciever distribution given the current state of the system'''
        
        # Get the iterval were the midpoint of stretched iterval belongs to:
        (a, b), (_, _), (a_actual, b_actual) = self.find_in_midpoints(mode = "start_stretch", split_point = 0.5)
        # Find the actual point that correspoind to stretched midpoint
        new_actual = ((0.5 - a) * b_actual + (b - 0.5) * a_actual) / (b - a)

        # Stretch the recieved values
        new_midpoint_positions = self.midpoint_positions.copy()
        new_midpoint_positions[self.midpoint_positions[:,1] < 0.5, 1] = self.midpoint_positions[self.midpoint_positions[:,1] < 0.5, 1] * (self.channel.p if self.rx_bit else self.channel.q) / 0.5
        new_midpoint_positions[self.midpoint_positions[:,1] > 0.5, 1] = (self.midpoint_positions[self.midpoint_positions[:,1] > 0.5, 1] - 0.5) * (self.channel.q if self.rx_bit else self.channel.p) / 0.5 + (self.channel.p if self.rx_bit else self.channel.q)
        new_midpoint_positions = np.append(new_midpoint_positions, [[new_actual, 0.5 * (self.channel.p if self.rx_bit else self.channel.q) / 0.5]], axis = 0)
        self.midpoint_positions = new_midpoint_positions

        self.log(f"Finished update_reciever_dist:\n midpoints : {self.midpoint_positions}")

    def tx(self):        
        
        self.log("POSITIONS TX", np.where(self.midpoint_positions[:, 1] == 1.0))
        self.log("POSITIONS TX", np.where(self.midpoint_positions[:, 1] == 0.0))

        self.log(self.midpoint_positions)
        self.log("Current seq", self.current_sequence)
        self.log("Times", self.time_in_queue)

        # Check if reciever has decoded something and adjust the current sending message appropriately
        if self.rx_decoded == 1 or self.rx_decoded == 0:
            self.current_sequence = self.current_sequence[1:]
            self.time_in_queue = self.time_in_queue[1:]

        # Find the message point
        n = self.current_sequence.shape[0]
        M = 2**n # number of possible sequences in current transmission
        while 1/M > self.d0:
            if self.get_additional_bit():
                M*=2
                n+=1
            else:
                break
        
        number = np.dot(np.flip(self.current_sequence), 2**np.arange(n))
        message_point = (number + 1/2) / M
        # Figure out the transmit bit
        # 1) Find the interval that contains the message point
        # 2) Figure out the coordinate of the message point on the current interval 
        (a_actual, b_actual), (_, _), (a_stretch, b_stretch) = self.find_in_midpoints(mode = "start_actual", split_point = message_point)
        message_point_stretched = ((message_point - a_actual) * b_stretch + (b_actual - message_point) * a_stretch) / (b_actual - a_actual)

        if message_point_stretched >= 0.5:
            self.tx_bit = 1
        elif message_point_stretched <= 0.5:
            self.tx_bit = 0

        self.log(f"Finished Tx:\n midpoints : {self.midpoint_positions} \n Current sequence: {self.current_sequence} \n Sending bit {self.tx_bit} \n\n")


    def rx(self):

        # self.log("POSITIONS", np.where(self.midpoint_positions[:, 1] == 1))
        # self.log("POSITIONS", np.where(self.midpoint_positions[:, 1] == 0))


        # Reciveve bit throught the channel
        self.rx_bit = self.channel.rx(self.tx_bit)
        # Perform checks on the input distribution
        self.update_reciever_dist()
        # Check if one of the bits can be detected with the specified probability of error
        # There is a number in self.midpoint_positions should correspond to the true midpoint of
        # our interval. We test for our conditions and if the probability is large enough,
        # we remove this midpoint and rescale our midpoints accordingly 

        # Find a number that corresponds to
        try:
            (a_actual, b_actual), (_, _), (a_stretch, b_stretch) = self.find_in_midpoints(mode = "start_actual", split_point = 0.5)
            mid_stretch = ((0.5 - a_actual) * b_stretch + (b_actual - 0.5) * a_stretch) / (b_actual - a_actual)
        except:
            raise ValueError(f"Error while finding a midpoint: {self.midpoint_positions}")
        
        if mid_stretch >= 1 - self.Pe:
            self.rx_bitstream = np.append(self.rx_bitstream, 0)
            self.rx_decoded = 0
            # remove all the positions where the actual value is higher than 0.5
            self.midpoint_positions = self.midpoint_positions[self.midpoint_positions[:, 0] < 0.5]
            # Scale the rest of the numbers appropriately
            self.midpoint_positions *= np.array([[2.0, 1 / mid_stretch]]) # multiply actual values by 2 and the stretched values by 1 / {stretched position of 0.5}

        elif mid_stretch < self.Pe:
            self.rx_bitstream = np.append(self.rx_bitstream, 1)
            self.rx_decoded = 1
            # remove all the positions where the actual value is less than 0.5
            self.midpoint_positions = self.midpoint_positions[self.midpoint_positions[:, 0] > 0.5]
            # Scale the rest of the numbers appropriately
            self.midpoint_positions = (self.midpoint_positions - np.array([[0.5, mid_stretch]])) * np.array([[2.0, 1 / (1 - mid_stretch)]])
        else:
            self.rx_decoded = None

        # Add [0, 0] or [1, 1] if they're missing after restretching
        if 0.0 not in self.midpoint_positions[:, 0]:
            # raise ValueError(f"AAAAAAAAA {self.midpoint_positions}")
            self.midpoint_positions = np.append(self.midpoint_positions, [[0.0, 0.0]], axis=0)
        if 1.0 not in self.midpoint_positions[:, 0]:
            # raise ValueError(f"AAAAAAAAA {self.midpoint_positions}")
            self.midpoint_positions = np.append(self.midpoint_positions, [[1.0, 1.0]], axis=0)

        if self.rx_decoded is not None:
            self.N_average.update(self.time_in_queue[0])

        self.time_in_queue = self.time_in_queue + 1

        self.log(f"Finished Rx:\n midpoints : {self.midpoint_positions} \n Recieved sequence: {self.rx_bitstream} \n\n")

def test_error_exponent(Pe_sweep = np.logspace(-1, -4, 16)):
    '''Run Horstein scheeme for various probabilities of error'''
    r = Parallel(n_jobs=16)(delayed(lambda x, i: simulate_horstein(Pe = x, pos = i))(Pe, i) for i, Pe in enumerate(Pe_sweep))
    return r

def simulate_horstein(channel_p = 0.1, l = 1000, Pe = 0.001, d0 = 0.000001, pos = 0, enable = False):
    print(f"position {pos}")
    if d0 == None:
        d0 = Pe / 10
    channel = BSC(channel_p)
    print(f"Capacity: {channel.capacity()}")
    horst = Horstein(channel=channel, tx_bitstream=np.random.randint(0, 2, l), Pe = Pe, enable=enable)
    
    it = 0
    # progress_bar = tqdm.tqdm(total=l, desc="Progress", leave=True, position=pos)
    while horst.rx_bitstream.size < l:
        it += 1
        # progress_bar.n = horst.rx_bitstream.size
        # progress_bar.refresh()
        # progress_bar.set_description("Iteration {}".format(it))
        horst.tx()
        horst.rx()

    # progress_bar.close()
    # print(f"Were sending: {horst.tx_bitstream[:l]}")
    # print(f"Recieved: {horst.rx_bitstream[:l]}")
    errors = np.sum(np.abs(horst.tx_bitstream[:l] - horst.rx_bitstream[:l]))
    # print("Decoded wrong: ", np.sum(np.abs(horst.tx_bitstream[:l] - horst.rx_bitstream[:l])))
    return {
        "sent": horst.tx_bitstream[:l],
        "recieved": horst.tx_bitstream[:l],
        "error": errors / l,
        "N": horst.N_average.avg,
        "Pe": Pe,
        "effecitve_rate": l / it
    }


if __name__ == "__main__":
    """Perform testin of some of the funcitons"""
    np.set_printoptions(precision=25)
    # horst.midpoint_positions = np.append(horst.midpoint_positions, [[0.5, 0.8], [0.4, 0.2]], axis = 0)
    # horst.tx_bit = 0
    # horst.update_reciever_dist()
    # horst.update_reciever_dist()
    # horst.update_reciever_dist()
    # out = simulate_horstein(enable=True)
    # print(out)
    r = test_error_exponent()
    print(r)
    serialized_data = jsonpickle.encode(r, unpicklable=False)

    file_path = 'r.json'
    with open(file_path, 'w') as outfile:
        outfile.write(serialized_data)

    with open(file_path, 'r') as infile:
        json_data = infile.read()

    # # Deserialize using jsonpickle
    # data = jsonpickle.decode(json_data)

    fig, ax1 = plt.subplots()

    ax1.plot(-np.log10([a['Pe'] for a in r]), [a['N'] for a in r], label = "N")
    ax2 = ax1.twinx()
    ax2.plot(-np.log10([a['Pe'] for a in r]), [a['error'] for a in r], label = "error", color = "r")
    fig1, ax3 = plt.subplots()
    ax3.plot(-np.log10([a['Pe'] for a in r]), [a['effecitve_rate'] for a in r], label = "rate", color = "g")
    
    ax2.set_yscale('log')


    ax1.legend()
    ax2.legend()
    ax3.legend()


    # plt.xscale('log')
    ax1.set_xlabel("1/Pe, log10")
    ax1.set_ylabel("Data")
    ax1.grid(True)
    fig.savefig(f"result3.png")






