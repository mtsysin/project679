"""Contains all of the code needed to simulate the channel in general"""
import numpy as np

class System:
    def __init__(self, channel, tx_size = 1024) -> None:
        self.tx_size = tx_size
        self.transmit: np.ndarray = np.random.randint(0, 2, tx_size)
        self.channel = channel

    def simulate(self):
        pass
        

# class AvgCounter:
#     def __init__(self) -> None:
#         self.count

#     def increment(self):

#     def calcualate(self):
    


    

class BSC:
    def __init__(self, p) -> None:
        """Defines a container for BSC with error probability p"""
        self.p = p
        self.q = 1-p
    
    def rx(self, tx: bool):
        return (not tx) if np.random.rand() < self.p else tx

class Horstein:
    def __init__(self, tx_bitstream = np.array([]), rx_bitstream = np.array([]), channel: BSC = None, Pe = 0.01, d0 = 0.01) -> None:
        self.channel = channel
        # initialize the interval
        self.midpoint_positions: np.ndarray = np.array([[0.0, 0.0], [1.0, 1.0]]) # array of: actual_value, position after stretching
        self.Pe = Pe
        self.current_sequence = np.array([])
        self.tx_bitstream: np.ndarray = tx_bitstream
        self.rx_bitstream: np.ndarray = rx_bitstream
        self.rx_bit = None
        self.tx_bit = None
        self.d0 = d0
        self.tx_bitstream_idx = 0
        self.rx_decoded = None

        # Evaluation metrics
        self.N_average = 0
        self.N_count = 0
        self.N_running = 0

    def get_additional_bit(self):
        if self.tx_bitstream_idx < self.tx_bitstream.size:
            extra_bit = self.tx_bitstream[self.tx_bitstream_idx]
            self.tx_bitstream_idx+=1
            self.current_sequence = np.append(self.current_sequence, extra_bit)
            return True
        return False

    def find_in_midpoints(self, mode = "start_actual", split_point = 0.5):
        if mode == "start_stretch":
            row, row_second = 1, 0
        elif mode == "start_actual":
            row, row_second = 0, 1

        midpoints_base = self.midpoint_positions[:, row]
        sorted_midpoints_base = np.sort(midpoints_base)
        idx_split = np.searchsorted(sorted_midpoints_base, split_point, side='left', sorter=None)
        a_base, b_base = sorted_midpoints_base[idx_split-1], sorted_midpoints_base[idx_split]
        a_ind, b_ind = np.where(midpoints_base == a_base), np.where(midpoints_base == b_base)
        try:
            a_second, b_second = self.midpoint_positions[a_ind[0].item(), row_second], self.midpoint_positions[b_ind[0].item(), row_second]
        except:
            raise ValueError(f"There are none or multiple values for a_base, b_base: {self.midpoint_positions}, {a_base, b_base}")

        return (a_base, b_base), (a_ind, b_ind), (a_second, b_second)


    def update_reciever_dist(self):
        
        (a, b), (_, _), (a_actual, b_actual) = self.find_in_midpoints(mode = "start_stretch", split_point = 0.5)
        new_actual = ((0.5 - a) * b_actual + (b - 0.5) * a_actual) / (b - a)

        # Stretch the recieved values
        new_midpoint_positions = self.midpoint_positions.copy()
        new_midpoint_positions[self.midpoint_positions[:,1] < 0.5, 1] = self.midpoint_positions[self.midpoint_positions[:,1] < 0.5, 1] * (self.channel.p if self.rx_bit else self.channel.q) / 0.5
        new_midpoint_positions[self.midpoint_positions[:,1] > 0.5, 1] = (self.midpoint_positions[self.midpoint_positions[:,1] > 0.5, 1] - 0.5) * (self.channel.q if self.rx_bit else self.channel.p) / 0.5 + (self.channel.p if self.rx_bit else self.channel.q)
        new_midpoint_positions = np.append(new_midpoint_positions, [[new_actual, 0.5 * (self.channel.p if self.rx_bit else self.channel.q) / 0.5]], axis = 0)
        self.midpoint_positions = new_midpoint_positions

        print(f"Finished update_reciever_dist:\n midpoints : {self.midpoint_positions}")

    def tx(self):        
        
        print("POSITIONS TX", np.where(self.midpoint_positions[:, 1] == 1.0))
        print("POSITIONS TX", np.where(self.midpoint_positions[:, 1] == 0.0))

        print(self.midpoint_positions)

        # Check if reciever has decoded something and adjust the mending message appropriately
        if self.rx_decoded == 1 or self.rx_decoded == 0:
            self.current_sequence = self.current_sequence[1:]

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

        self.N_running += 1
        print(f"Finished Tx:\n midpoints : {self.midpoint_positions} \n Current sequence: {self.current_sequence} \n Sending bit {self.tx_bit} \n\n")


    def rx(self):

        print("POSITIONS", np.where(self.midpoint_positions[:, 1] == 1))
        print("POSITIONS", np.where(self.midpoint_positions[:, 1] == 0))


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
            self.N_average = (self.N_average * self.N_count + self.N_running) / (self.N_count + 1)
            self.N_count += 1

        print(f"Finished Rx:\n midpoints : {self.midpoint_positions} \n Recieved sequence: {self.rx_bitstream} \n\n")

        
if __name__ == "__main__":
    """Perform testin of some of the funcitons"""
    np.set_printoptions(precision=25)
    LEN = 4096
    horst = Horstein(channel=BSC(0.2), tx_bitstream=np.random.randint(0, 2, LEN))
    # horst.midpoint_positions = np.append(horst.midpoint_positions, [[0.5, 0.8], [0.4, 0.2]], axis = 0)
    # horst.tx_bit = 0
    # horst.update_reciever_dist()
    # horst.update_reciever_dist()
    # horst.update_reciever_dist()

    i = 0
    # for i in range(100):
    while horst.rx_bitstream.size < LEN:
        i += 1
        print(i, "###############################")
        horst.tx()
        horst.rx()


    print(f"Were sending: {horst.tx_bitstream[:LEN]}")
    print(f"Recieved: {horst.rx_bitstream[:LEN]}")

    print(np.sum(np.abs(horst.tx_bitstream[:LEN] - horst.rx_bitstream[:LEN])))



