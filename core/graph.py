import random
import math
from core.node import Node
from core.distance import l2_distance
def assign_layer(M:int)->int:
    mL=1/math.log(M)
    r=random.random()
    while r==0.0:
        r=random.random()
    layer=math.floor(-math.log(r)*mL)
    return layer



class HNSWGraph:
    def __init__(self, M=16):
        self.nodes={}
        self.entry_point=None
        self.M=M
        self.max_layer=-1

    def insert(self,id,vector):
        layer=assign_layer(self.M)
        node=Node(id=id,max_layer=layer,vector=vector)
        if len(self.nodes)==0:
            self.entry_point=node
            self.max_layer=layer
        self.nodes[id]=node

    def search_layer_greedy(self,query_vector,entry_id,layer):
        current=entry_id
        while True:
            neighbor=self.nodes[current].neighbors.get(layer,[])
            closer_found=False
            for n in neighbor:
                if l2_distance(self.nodes[n].vector,query_vector)< l2_distance(self.nodes[current].vector,query_vector):
                    current=n
                    closer_found=True
            if not closer_found:
                break

        return current       
