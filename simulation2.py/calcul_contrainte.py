"""Ce code sert à construire la fonction qui va coder la matrice de contrainte en fonction de l'espace et du temps."""


#-------------------------------------#
#Définitions des constantes
#-------------------------------------#

Q = 0.04 #Le débit volumique d'injection de fluide

lambdaa = 16E9 #coefficient de Larmé drainé

lambdaa_u = 37E9 #coefficient de Larmé non drainé

mu = 16E9 #Module de cisaillement

D = 0.1 #diffusivité hydraulique

alpha = 0.8 #coefficient de Biot

#-------------------------------------#
#Définitions de fonctions utiles
#-------------------------------------#
import numpy as np
import scipy as sp

def erfc(x):
    #Définition de la fonction erfc
    return (2/np.sqrt(np.pi))*sp.integrate.quad(lambda y : np.exp(-y**2),x,np.inf) 

def g(x):
    #Définition de la fonction g
    return (2/np.pi)*sp.integrate.quad(lambda y : np.exp(-y**2),0,x/2) - (x/np.pi)*np.exp(-x**2/4)

