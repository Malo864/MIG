# ---------------------------------------------------------------------
# 1.Définition du Volume d'espace 
# ---------------------------------------------------------------------
import numpy as np

dx, dy, dz = 0.001 , 0.001 , 0.001 #paramètres infinitésimaux (précision spatiale entre deux cases d'un tableau)

n = 100000 # Nombre de colonne d'unn tableau 

X , Y , Z = n*dx , n*dy, n*dz #Dimensions du volume

Volume = X*Y*Z #Calcul du volume occupé

# ---------------------------------------------------------------------
# 2.Tirage au sort du nombre de failles
# ---------------------------------------------------------------------

'''Rappelons-nous tout d'abord de la loi de Gutenberg-Richter, on a une décroissance
exponentielle du nombre de faille en fonction de leur taille. Ici, l'espace
considéré étant un cube de largeur 100m avec une discrétisation de 1mm
contentons-nous de failles d'une largeur entre 1m et 1 cm.
La loi de Gutenberg-Richter selon la taille de la faille donne
 alors que on a une répartition en une puissance de 1/L. 
 (Voir les Notes à ce propos dans le dossier création de faille.)'''

alpha = 1.35 #paramètre d'évolution caractéristique de la loi de Gutenberg-Richter.
N0 = 3.1E3 #paramètre caractéristique de la sismicité (choisi ici pour avoir une
#faille de taille 10m)


def N(L,alph = alpha,N_0 = N0):
    """Cette fonction calcule le nombre de faille à partir de leur longueur."""
    return np.floor(N_0 * L**(-alph))

N_10m , N_1m, N_10cm ,  N_1cm = N(10), N(1) , N(0.1) , N(0.01)

# ---------------------------------------------------------------------
# 3.Tirage au sort de la position et de l'orientation des failles
# ---------------------------------------------------------------------
import random as rd
def tirage():
    """Cette fonction renvoie aléatoirement une position dans le cube de volume
    et une orientation sous la forme d'un triplet (x,y,z) et d'un vecteur normal."""
    x , y , z = rd.randint(0,n)*dx , rd.randint(0,n)*dy , rd.randint(0,n)*dz
    n = (rd.random()*2-1),(rd.random()*2-1),(rd.random()*2-1)
    n=n/np.sqrt(n[0]**2+n[1]**2+n[2]**2)
    return (x,y,z) , n

L_faille =[]

for i in range(N_10m):
    pos,n=tirage()
    L_faille.append(10,pos,n)
for i in range(N_1m):
    pos,n=tirage()
    L_faille.append(1,pos,n)
for i in range(N_10cm):
    pos,n=tirage()
    L_faille.append(0.1,pos,n)
for i in range(N_1cm):
    pos,n=tirage()
    L_faille.append(0.01,pos,n)

#Dans L_faille on a maintenant la liste des failles, avec leur position, leur orientation
#et leur longueur.

