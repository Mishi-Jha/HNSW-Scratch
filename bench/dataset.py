import numpy as np
def generate_random_dataset(n, dim, seed=None) -> np.ndarray:
    rng=np.random.default_rng(seed)
    array=rng.random((n,dim))
    return array
        
