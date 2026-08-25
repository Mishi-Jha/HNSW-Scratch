import numpy as np
from numpy import linalg as la
def l2_distance(a,b):
    return np.linalg.norm(a-b)

def cosine_distance(a,b):
    dotProduct=np.dot(a, b)
    mag_a=np.linalg.norm(a)
    mag_b=np.linalg.norm(b)
    similarity=dotProduct/(mag_a*mag_b)
    return 1-similarity 