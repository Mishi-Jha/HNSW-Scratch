import random
import math
import heapq
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
    def __init__(self, M=16,ef_construction=100):
        self.nodes={}
        self.entry_point=None
        self.M=M
        self.ef_construction=ef_construction
        self.max_layer=-1

    def insert(self,id,vector):
        layer=assign_layer(self.M)
        node=Node(id=id,max_layer=layer,vector=vector)
        if len(self.nodes)==0:
            self.entry_point=node
            self.max_layer=layer
            self.nodes[id]=node
            return
        self.nodes[id]=node
        current_entry=self.entry_point.id
        n_layer=self.max_layer
        while node.max_layer<n_layer:
            current_entry=self.search_layer_greedy(node.vector,current_entry,n_layer)
            n_layer-=1

        n_min_layer=min(node.max_layer,self.max_layer)    
        while n_min_layer>=0:
            candidates_list=self.search_layer(node.vector,current_entry,n_min_layer,self.ef_construction)
            candidates_list=candidates_list[:self.M]
            for cand_id,dist in candidates_list:
                node.neighbors.setdefault(n_min_layer,[]).append(cand_id)
                self.nodes[cand_id].neighbors.setdefault(n_min_layer,[]).append(node.id)
            if candidates_list:
                current_entry=candidates_list[0][0]    
            n_min_layer-=1
        if node.max_layer>self.max_layer:
            self.max_layer=node.max_layer
            self.entry_point=node


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

    def search_layer(self, query_vector, entry_id, layer, ef):
        candidates=[]
        results=[]
        visited=set()
        distance=l2_distance(query_vector,self.nodes[entry_id].vector)
        heapq.heappush(candidates,(distance,entry_id))
        heapq.heappush(results,(-distance,entry_id))
        visited.add(entry_id)
        while candidates:
            dist,current=heapq.heappop(candidates)
            if len(results)==ef and dist>-results[0][0]:
                break
            layers=self.nodes[current].neighbors.get(layer,[])
            for l in layers:
                if l in visited:
                    continue
                else:
                    visited.add(l)
                    d=l2_distance(query_vector,self.nodes[l].vector) 
                    heapq.heappush(candidates,(d,l))
                    heapq.heappush(results,(-d,l))
                    if len(results)>ef:
                        heapq.heappop(results)
        final=[]
        for neg_dist,node_id in results:
            final.append((node_id,-neg_dist))
        sorted_final=sorted(final,key=lambda pair:pair[1])
        return sorted_final