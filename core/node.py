from dataclasses import dataclass, field

@dataclass
class Node:
    id:int
    max_layer:int
    neighbors:dict=field(default_factory=dict)