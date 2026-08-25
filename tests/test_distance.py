import pytest
import numpy as np
from core.distance import l2_distance, cosine_distance
def test_l2_distance():
    a=np.array([1,0])
    b=np.array([0,1])
    assert l2_distance(a,a)==0
    assert l2_distance(a,b)==np.sqrt(2)

def test_cosine_distance():
    a=np.array([0,0])
    b=np.array([0,0])
    assert np.isnan(cosine_distance(a,b))