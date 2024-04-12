'''
Porblem 1 code
Simulating mimo system ergodic capacity
'''

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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


class Simulator:
    def __init__(self, 
                 Mt  = 2, 
                 Mr = 2, 
                 num_iterations = 1000000,
                 snr_sweep = np.logspace(-5/10, 40/10, 30, base=10)
    ) -> None:
        self.Mt = Mt
        self.Mr = Mr
        self.num_iterations = num_iterations
        self.snr_sweep = snr_sweep

    def simulate_snr(self, snr):
        
        C = Average()
        for it in range(self.num_iterations):
            # Create an instance of H matrix
            H = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(self.Mr, self.Mt, 2)).view(np.complex128)[:, :, 0]
            _, d, _ = np.linalg.svd(H, full_matrices=False)
            gamma = self.do_waterfilling(d, snr)
            C.update(np.sum(np.log2(1 + snr * d**2 * gamma)))


        return C.avg

    def do_waterfilling(self, d, snr):
        '''Find the values of gamma that sum up to 1 and maximize sum_1^M(log2(1 + snr * dl^2*gamma_l))'''
        M = len(d)
        for t in range(1, M+1):
            mu = 1/(M - t + 1) * (1 + 1/snr * np.sum(1/d[:M-t+1]**2))
            gamma = mu - 1/snr * 1/d**2
            gamma[M-t+1:] = 0
            if gamma[M - t] >= 0:
                return gamma
            
    def generate_sweep(self):
        r = Parallel(n_jobs=20)(delayed(lambda x: self.simulate_snr(x))(snr) for snr in self.snr_sweep)
        print(r)
        # return np.array(list(map(lambda x: self.simulate_snr(x), self.snr_sweep)))
        

if __name__ == "__main__":
    sim = Simulator()
    plt.figure()
    db_scale = 10 * np.log10(sim.snr_sweep)
    dims = [(1, 1), (2, 2), (4, 4)]
    for dim in dims:
        sim.Mr, sim.Mt = dim
        plt.plot(db_scale, sim.generate_sweep(), label = str(dim))

    plt.legend()
    plt.xlabel("SNR, db")
    plt.ylabel("Capacity, bits/s/hz")
    plt.grid(True)
    plt.savefig(f"result{dims}.png")
