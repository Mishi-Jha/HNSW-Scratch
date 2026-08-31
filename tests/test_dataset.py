import numpy as np
from bench.dataset import generate_random_dataset

def test_generate_random_dataset_reproducible():
    a = generate_random_dataset(5, 3, seed=42)
    b = generate_random_dataset(5, 3, seed=42)
    assert np.array_equal(a, b)

def test_generate_random_dataset_shape():
    a = generate_random_dataset(10, 4, seed=1)
    assert a.shape == (10, 4)