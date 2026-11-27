import numpy as np
import random as rd
n = (rd.random()*2-1),(rd.random()*2-1),(rd.random()*2-1)
n=n/np.sqrt(n[0]**2+n[1]**2+n[2]**2)
print(n)