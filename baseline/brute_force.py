import numpy as np
from numpy import linalg as la
def brute_force_knn(query, dataset, k)-> list[tuple[int,float]]:
    diff=dataset-query
    distance=la.norm(diff,axis=1)
    idx=np.argpartition(distance,k)[:k]
    sorted_order=np.argsort(distance[idx])
    sorted_idx=idx[sorted_order]
    return list(zip(sorted_idx,distance[sorted_idx]))