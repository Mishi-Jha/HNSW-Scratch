from dataclasses import dataclass, field
import numpy as np
@dataclass
class Node:
    id:int
    max_layer:int
    vector: np.ndarray
    neighbors:dict=field(default_factory=dict)