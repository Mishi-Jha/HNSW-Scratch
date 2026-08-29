import random
import math
def assign_layer(M:int)->int:
    mL=1/math.log(M)
    r=random.random()
    while r==0.0:
        r=random.random()
    layer=math.floor(-math.log(r)*mL)
    return layer