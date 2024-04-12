'''
Porblem 1 code
Simulating mimo system ergodic capacity
'''

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class SimulatorBPSK:
    def __init__(self, 
                 snr_sweep = np.logspace(-5/10, 15/10, 30, base=10),
                 Mt  = 2, 
                 Mr = 2, 
                 num_iterations = 1000000
                 ) -> None:
        self.snr_sweep = snr_sweep
        self.Mt = Mt
        self.Mr = Mr
        self.num_iterations = num_iterations

    def transmit_symbol_simple(self, symbol):
        s = np.sqrt(self.snr) if symbol else -np.sqrt(self.snr)
        noise = np.random.normal(loc = 0, scale=1/np.sqrt(2))
        y = s + noise
        return int(y > 0)
    
    def make_mrt_mrc(self, H):
        '''Retunrs beamformer f and combiner z for mrt/mrc'''
        U, d, Vh  = np.linalg.svd(H, full_matrices=False)

        z = U[:,0:1]
        f = Vh[0:1,:].T.conj()

        return z, f
    

    def make_sdc_sdt(self, H):
        '''Retunrs beamformer f and combiner z for sdt/sdc'''
        H_abs = np.abs(H)
        k, l = np.unravel_index(np.argmax(H_abs, axis=None), H_abs.shape)

        # print(H_abs, k, l)

        z = np.zeros((self.Mr, 1))
        f = np.zeros((self.Mt, 1))

        z[k, 0] = 1
        f[l, 0] = 1

        return z, f


    def run(self, snr, mode = "mrc"):
        print(f"start simulation for {mode}, {snr}")

        err = 0
        for it in range(self.num_iterations):
            s = np.sqrt(snr) if np.random.rand() > 0.5 else -np.sqrt(snr)
            if mode == "simple":
                n = np.random.normal(loc = 0, scale=1/np.sqrt(2))
                y = s + n
                err += int(y * s < 0)
                continue

            # Create Rayleigh fading matrix
            H = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(self.Mr, self.Mt, 2)).view(np.complex128)[:, :, 0]
            if mode == "mrc":
                z_opt, f_opt = self.make_mrt_mrc(H)
            elif mode == "sdc":
                z_opt, f_opt = self.make_sdc_sdt(H)
            else:
                raise ValueError("wrong mode")
            # Make noise vector
            n = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(self.Mr, 2)).view(np.complex128)
            # print(n)
            n_eff = z_opt.T.conj() @ n

            # print(z_opt, f_opt)


            h_eff = z_opt.T.conj() @ H @ f_opt
            if mode == "mrt":
                assert h_eff >= 0

            # Can rotate coordiantes
            h_eff = np.abs(h_eff)

            # print(h_eff)
            
            y = h_eff * s + n_eff

            err += int(y * s < 0)

            # print(y)
            # return
        
        return err/self.num_iterations
    
    def run_sweep(self, mode = 'mrc'):
        r = Parallel(n_jobs=40)(delayed(lambda x: self.run(x, mode = mode))(snr) for snr in self.snr_sweep)
        print(r)
        return r

if __name__ == "__main__":
    sim = SimulatorBPSK()

    # sim.run(snr=5, mode = "mrc")

    plt.figure()
    db_scale = 10 * np.log10(sim.snr_sweep)
    modes = ['simple', 'mrc', 'sdc']
    for mode in modes:
        plt.plot(db_scale, sim.run_sweep(mode), label = str(mode))

    plt.legend()
    plt.xlabel("SNR, db")
    plt.ylabel("Pe")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f"result{modes}.png")