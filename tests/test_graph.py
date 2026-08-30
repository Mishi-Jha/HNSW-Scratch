import pytest
import numpy as np
from core.graph import HNSWGraph
from core.node import Node

def test_search_layer_greedy():
    g = HNSWGraph()
    g.nodes[0] = Node(id=0, max_layer=0, vector=np.array([0,0]), neighbors={0: [1]})
    g.nodes[1] = Node(id=1, max_layer=0, vector=np.array([5,5]), neighbors={0: [0, 2]})
    g.nodes[2] = Node(id=2, max_layer=0, vector=np.array([10,10]), neighbors={0: [1]})

    query = np.array([9,9])
    result = g.search_layer_greedy(query, entry_id=0, layer=0)

    assert result == 2