import numpy as np
import random as rd
def N(L,alph = 1.34,N_0 = 100):
    """Cette fonction calcule le nombre de faille à partir de leur longueur."""
    return int(N_0 * L**(-alph))

L_longueurs = [0.1,0.2,0.5,1,2,3,5,7,10,20]
N_longueurs = np.vectorize(N)(L_longueurs)
print(N_longueurs)