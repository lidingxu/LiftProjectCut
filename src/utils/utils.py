import numpy as np

class Options:
    def __init__(self, sepamethod = "s", seed = 0, verbose = True, device = "cuda", dtype = 64, spxinit = True, debug = False):
        self.sepamethod = sepamethod
        self.seed = seed
        self.verbose = verbose
        self.spxinit = spxinit
        self.debug = debug
        self.device = device
        self.dtype = dtype

def setSeed(seed):
    np.random.seed(seed)

def choice(population, size, p):
    probs = np.array(p)
    probs = probs / probs.sum()
    probs = None
    print(len(population), len(p), size)
    return np.random.choice(population, size, replace=False, p=probs)
