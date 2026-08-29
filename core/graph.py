import random
import math
from core.node import Node
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


