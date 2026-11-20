import scipy as sp
import numpy as np
def f(x):
    return np.exp(-x**2)/2

print(sp.integrate.quad(f,0,np.inf))