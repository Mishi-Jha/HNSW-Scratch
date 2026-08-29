import pytest
import numpy as np
from baseline.brute_force import brute_force_knn
def test_brute_force_knn():
    dataset=np.array([[1,0],[0,1],[5,5]])
    query=np.array([1,0])
    assert brute_force_knn(query=query,dataset=dataset,k=1)==[(0,0.0)]
    result= brute_force_knn(query=query,dataset=dataset,k=2)
    expected=[(0,0.0),(1,np.sqrt(2))]
    assert len(result)==len(expected)
    for(r_idx,r_dist),(e_idx,e_dist) in zip(result,expected):
        assert r_idx==e_idx
        assert np.isclose(r_dist,e_dist)

    