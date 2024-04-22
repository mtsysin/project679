import numpy as np
from scipy import special


def Hb(p):
    return p * np.log2(1/p) + (1-p) * np.log2(1/(1-p))

def Q(x, *args, **kwargs):
    return 0.5 - 0.5*special.erf(x/np.sqrt(2))