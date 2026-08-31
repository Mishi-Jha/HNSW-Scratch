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

def test_search_layer():
    g = HNSWGraph()
    g.nodes[0] = Node(id=0, max_layer=0, vector=np.array([0,0]), neighbors={0: [1]})
    g.nodes[1] = Node(id=1, max_layer=0, vector=np.array([5,5]), neighbors={0: [0, 2]})
    g.nodes[2] = Node(id=2, max_layer=0, vector=np.array([10,10]), neighbors={0: [1, 3]})
    g.nodes[3] = Node(id=3, max_layer=0, vector=np.array([20,20]), neighbors={0: [2]})

    query = np.array([9,9])
    result = g.search_layer(query, entry_id=0, layer=0, ef=2)

    result_ids = [r[0] for r in result]
    assert result_ids == [2, 1]    

def test_insert_creates_bidirectional_edges():
    g = HNSWGraph(M=2, ef_construction=10)
    g.insert(0, np.array([0, 0]))
    g.insert(1, np.array([1, 1]))
    g.insert(2, np.array([10, 10]))

    assert set(g.nodes.keys()) == {0, 1, 2}

    for node_id, node in g.nodes.items():
        for layer, neighbor_ids in node.neighbors.items():
            for neighbor_id in neighbor_ids:    
                assert node_id in g.nodes[neighbor_id].neighbors.get(layer, []), \
                    f"node {node_id} lists {neighbor_id} at layer {layer}, but not reciprocated"