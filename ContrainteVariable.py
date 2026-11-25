#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. STRUCTURES DE DONNÉES (CONTENEURS)
# =========================================================================
class ParamMecDim:
    """Regroupe les paramètres mécaniques réels (dimensionnés, unités SI)."""
    pass

class NdParamMec:
    """Regroupe les paramètres mécaniques adimensionnés utilisés par le solveur."""
    pass

class ParamComp:
    """Regroupe les paramètres numériques du solveur."""
    pass

class PressionFunction:
    "Regroupe les paramètres de la fonction pression"
    def __init__(self,nom):
        self.nom=nom

# =========================================================================
# 2. PARAMÈTRES PHYSIQUES DIMENSIONNÉS ET ADIMENSIONNEMENT EXPLICITE
# =========================================================================
# A. Paramètres réels de la faille et de la croûte terrestre (unités SI)
pd = ParamMecDim()
pd.a = 0.005          # Paramètre d'effet direct de la loi Rate-and-State (sans unité)
pd.b = 0.010          # Paramètre d'évolution d'état de la loi Rate-and-State (sans unité)
pd.dc = 2.0e-3        # Distance critique de glissement : 2 mm (m)
pd.sigma0 = 45.0e6     # Contrainte normale effective sur la faille : 45 MPa (Pa)
pd.mu = 27.0e9        # Module de cisaillement de la roche : 27 GPa (Pa)
pd.rho = 3000.0       # Masse volumique de la roche (kg/m^3)
pd.Vp = 1.0e-9        # Vitesse de chargement tectonique : ~3,15 cm/an (m/s)
pd.f0 = 0.6           # Coefficient de frottement de référence (-)

pd.L = 300.0          # Dimension caractéristique du patch de faille : 300 m (m) forme une faille
#pd.L = 230.0          #asismique


# Grandeurs intermédiaires dimensionnées déduites des constantes élastiques
pd.beta = np.sqrt(pd.mu / pd.rho)     # Célérité des ondes S : 3000 m/s (m/s)
pd.eta_dim = pd.mu / (2.0 * pd.beta)  # Amortissement radiatif dimensionné : 4.5e6 Pa.s/m
pd.k_dim = pd.mu / pd.L               # Raideur élastique effective par unité de surface : 9.0e7 Pa/m

# B. Calcul explicite des paramètres adimensionnés (identiques au script original)
pnd = NdParamMec()
pnd.alpha = pd.a / pd.b                                   # alpha = a / b = 0.5 (< 1 : affaiblissant avec la vitesse)
pnd.eta0 = (pd.eta_dim * pd.Vp) / (pd.b * pd.sigma0)    # eta_adim = (mu * Vp) / (2 * beta * b * sigma0) = 1.0e-8
pnd.k0 = (pd.k_dim * pd.dc) / (pd.b * pd.sigma0)        # kappa = (k_dim * dc) / (b * sigma0) = 0.4 (< 1 - alpha = 0.5)
pnd.beta=pd.f0/pd.b

# Affichage de contrôle dans la console
print(f"Paramètres adimensionnés : a = {pnd.alpha}, eta = {pnd.eta0:.1e}, k = {pnd.k0}")

# C. Paramètres de calcul pour l'intégration temporelle
pc = ParamComp()
pc.tol = 1.0E-10       # Tolérance d'erreur locale maximale autorisée
pc.nitrkmax = 30       # Nombre max de tentatives pour ajuster le pas de temps
pc.nitmax = 22000      # Nombre total d'itérations temporelles de la simulation
pc.hmin = 1.0E-12      # Pas de temps minimal autorisé (pendant les séismes)
pc.hmax = 1.0E10       # Pas de temps maximal autorisé (pendant les phases lentes)
pc.safe = 0.8          # Facteur de sécurité pour le recalcul du pas de temps

# D. Définition des différents paramètres des fonctions possibles pour la pression du fluide 
exp = PressionFunction("Exponentielle") # f(x)=A*exp(t/tau)
step = PressionFunction("Step")
lin = PressionFunction("Lineaire")
per = PressionFunction("Périodique")

lin.t0 = 400.0        # Instant de début de montée
lin.rate = 0.1e6      # Pente de montée (ex: 0.5 MPa par unité de temps)
lin.pmax = 20.0e6     # Plafond optionnel (mettre None pour une montée illimitée)

step.t0=400.0
step.tau=1
step.pmax=5.0e6

exp.t0 = 150.0        # Instant de début d'injection
exp.tau = 30.0        # Temps caractéristique de montée en pression
exp.pmax = 8.0e6      # Palier asymptotique de surpression (ex: 8 MPa < sigma0)

per.t0 = 150.0         # Instant de début des oscillations
per.T = 15.8           # Période d'oscillation (ex: 15.8 en adim ≈ 1 an)
per.pmax = 4.0e6       # Surpression maximale (oscille entre 0 et 4 MPa)

scenario=step


# =========================================================================
# 3. CONDITIONS INITIALES
# =========================================================================
v = 0.1        # Vitesse initiale adimensionnée V / Vp (soit V_init = 1e-10 m/s, patin freiné)
th = 1.0 / v   # Variable d'état initiale adimensionnée theta / (dc / Vp)
t = 0.0        # Temps initial adimensionné t / (dc / Vp)
h = 0.001      # Pas de temps initial adimensionné

# Passage en variables logarithmiques (garantit v > 0 et th > 0, stabilise le calcul)
phi = np.log(v)   # phi = ln(v)
nu = np.log(th)   # nu = ln(th)


# =========================================================================
# 4. ÉQUATIONS DIFFÉRENTIELLES DU MODÈLE (RATE-AND-STATE)
# =========================================================================
def frns(t,phi, nu,pnd,pd,PF):
    """Calcule d(phi)/dt : équilibre mécanique quasi-dynamique du patin-ressort."""
    #On commence par calculer sigma la contrainte
    sigma,dsigma=fun_sigma(t,pd,PF)
    F = -pnd.k0 * (np.exp(phi) - 1) + (np.exp(phi) - np.exp(-nu))*sigma-(pnd.beta+pnd.alpha*phi+nu)*dsigma
    F = F / (pnd.alpha*sigma + pnd.eta0 * np.exp(phi))
    return F

def grns(phi, nu):
    """Calcule d(nu)/dt : loi d'évolution de l'état (loi de vieillissement de Dieterich)."""
    return np.exp(-nu) - np.exp(phi)

def fun_sigma(t,pd,fonction):
    #Principe de Terzaghi : sigma_eff=sigma-p
    pressure=fun_pression(t,fonction)
    return 1-pressure[0]/pd.sigma0, -pressure[1]/pd.sigma0

def fun_pression(t,fonction=None):
    if fonction==None or t<=fonction.t0:
        return 0.0,0.0
    if fonction.nom=="Step":
        #Fonction de la forme p(x)=pmax*[1-cos(np.pi*(t-t0)/tau)]
        #p'(x)=(pmax*np.pi/fonction.tau)*sin(np.pi*(t-t0)/tau))
        if t>=fonction.tau+fonction.t0:
            return fonction.pmax,0
        return (fonction.pmax/2)*(1-np.cos(np.pi*(t-fonction.t0)/fonction.tau)),(fonction.pmax*np.pi/(fonction.tau*2)       )*np.sin(np.pi*(t-fonction.t0)/fonction.tau)
    if fonction.nom == "Lineaire":
        t0 = fonction.t0
        R = fonction.rate
        pmax = fonction.pmax
        dt_r = getattr(fonction, "dt_raccord", 0.05 * (pmax / R))
        dt_r = min(dt_r, 0.5 * (pmax / R))  # Sécurité pour que dt_r <= moitié de la montée
        t_fin = t0 + (pmax / R) + dt_r
        if t >= t_fin:
            return pmax, 0.0
        elif t < t0 + dt_r:
            dp = R * (t - t0) / dt_r
            p = 0.5 * R * ((t - t0) ** 2) / dt_r
            return p, dp
        elif t <= t_fin - dt_r:
            p = 0.5 * R * dt_r + R * (t - (t0 + dt_r))
            return p, R
        else:
            dp = R * (t_fin - t) / dt_r
            p = pmax - 0.5 * R * ((t_fin - t) ** 2) / dt_r
            return p, dp
    if fonction.nom == "Exponentielle":
        """
        Approche exponentielle lissée (classe C^1) vers un palier pmax :
        - Vaut 0 et dérivée nulle en t = t0
        - Tend asymptotiquement vers pmax avec le temps caractéristique tau
        Formule : p(t)  = pmax * (1 - exp(-(t - t0)/tau))^2
                  p'(t) = (2 * pmax / tau) * exp(-(t - t0)/tau) * (1 - exp(-(t - t0)/tau))
        """
        exp_term = np.exp(-(t - fonction.t0) / fonction.tau)
        one_minus_exp = 1.0 - exp_term
        p = fonction.pmax * (one_minus_exp ** 2)
        dp = (2.0 * fonction.pmax / fonction.tau) * exp_term * one_minus_exp
        return p, dp
    if fonction.nom == "Periodique":
        """
        Forçage périodique sinusoïdal lissé (classe C^1) démarrant en t0 :
        - Vaut 0 et dérivée nulle en t = t0
        - Oscille entre 0 et pmax (amplitude crête-à-crête = pmax) de période T
        Formule : p(t)  = 0.5 * pmax * (1 - cos(2*pi*(t - t0)/T))
                  p'(t) = (pi * pmax / T) * sin(2*pi*(t - t0)/T)
        """
        omega = 2.0 * np.pi / fonction.T
        phase = omega * (t - fonction.t0)
        return 0.5 * fonction.pmax * (1.0 - np.cos(phase)), (0.5 * fonction.pmax * omega) * np.sin(phase)

# step.t0=10.0
# step.tau=20.0
# step.pmax=50.0
# Q=[]
# for i in range(0,100,1):
#     Q.append(fun_pression(i,step))
# plt.plot(np.arange(0,100,1),Q)
# plt.show()


def rkf(t, phi, nu, h, pnd, pc, pd, PF=None): #où PF=PressionFunction
    """Intégrateur RKF45"""
    # Nœuds temporels de Butcher (indispensables car frns dépend de t)
    a2, a3, a4, a5, a6 = 1/4, 3/8, 12/13, 1.0, 1/2

    # Coefficients internes de Butcher
    c21 = 1/4
    c31, c32 = 3/32, 9/32
    c41, c42, c43 = 1932/2197, -7200/2197, 7296/2197
    c51, c52, c53, c54 = 439/216, -8, 3680/513, -845/4104
    c61, c62, c63, c64, c65 = -8/27, 2, -3544/2565, 1859/4104, -11/40

    # Poids d'erreur et de mise à jour
    r1, r3, r4, r5, r6 = 1/360, -128/4275, -2197/75240, 1/50, 2/55
    c1, c3, c4, c5, c6 = 25/216, 1408/2565, 2197/4104, -1/5, 2/55

    passtep = 0
    iterrk = 0

    h_old=h
    while passtep == 0 and iterrk <= pc.nitrkmax:
        iterrk += 1

        k1 = h * frns(t, phi, nu, pnd, pd, PF)
        l1 = h * grns(phi, nu)

        k2 = h * frns(t + a2*h, phi + c21*k1, nu + c21*l1, pnd, pd, PF)
        l2 = h * grns(phi + c21*k1, nu + c21*l1)

        dphi, dnu = c31*k1 + c32*k2, c31*l1 + c32*l2
        k3 = h * frns(t + a3*h, phi + dphi, nu + dnu, pnd, pd, PF)
        l3 = h * grns(phi + dphi, nu + dnu)

        dphi, dnu = c41*k1 + c42*k2 + c43*k3, c41*l1 + c42*l2 + c43*l3
        k4 = h * frns(t + a4*h, phi + dphi, nu + dnu, pnd, pd, PF)
        l4 = h * grns(phi + dphi, nu + dnu)

        dphi, dnu = c51*k1 + c52*k2 + c53*k3 + c54*k4, c51*l1 + c52*l2 + c53*l3 + c54*l4
        k5 = h * frns(t + a5*h, phi + dphi, nu + dnu, pnd, pd, PF)
        l5 = h * grns(phi + dphi, nu + dnu)

        dphi, dnu = (c61*k1 + c62*k2 + c63*k3 + c64*k4 + c65*k5,
                     c61*l1 + c62*l2 + c63*l3 + c64*l4 + c65*l5)
        k6 = h * frns(t + a6*h, phi + dphi, nu + dnu, pnd, pd, PF)
        l6 = h * grns(phi + dphi, nu + dnu)

        err = (r1*np.array([k1, l1]) + r3*np.array([k3, l3]) +
               r4*np.array([k4, l4]) + r5*np.array([k5, l5]) + r6*np.array([k6, l6]))
        errmax = np.max(np.abs(err))
        h_old=h
        if 0 <= errmax <= pc.tol:
            passtep = 1
            phi += c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6
            nu += c1*l1 + c3*l3 + c4*l4 + c5*l5 + c6*l6
            if errmax > 0:
                h = min(pc.safe * h * ((pc.tol / errmax) ** 0.25), pc.hmax)
        else :
            if errmax > pc.tol:
                h = max(pc.safe * h * ((pc.tol / errmax) ** 0.25), pc.hmin)
            else:
                h=pc.hmin
    return phi, nu, h,h_old





# =========================================================================
# 6. BOUCLE PRINCIPALE DE SIMULATION
# =========================================================================
T = np.array([t])
Phi = np.array([phi])
Nu = np.array([nu])

#sigma0=45.0e6
pas=[0]

def redim_h(t,h,function=None):
    # 1. Atterrir exactement sur step.t0 sans le dépasser d'un grand pas
    if function!=None:
        if t < function.t0 < t + h:
            return function.t0 - t
        # 2. Pendant la rampe, imposer au moins 20 points pour échantillonner la variation
        elif function.nom=="Step" and function.t0 <= t < function.t0 + function.tau:
            return min(h, function.tau / 50.0)
    return h

for iter in range(0, pc.nitmax, 1):
    h=redim_h(t,h,scenario)

    phi, nu, h, vrai_pas = rkf(t,phi, nu, h, pnd,pc,pd,scenario)
    t += vrai_pas
    pas.append(vrai_pas)
    T = np.append(T, [t])
    Phi = np.append(Phi, [phi])
    Nu = np.append(Nu, [nu])

# print(pas[:500])
# plt.plot(T,pas)
# plt.show()
# =========================================================================
# 7. VISUALISATION DES RÉSULTATS
# =========================================================================
# Tracé en variables adimensionnées (identique au script original)
plt.plot(T, np.log10(np.exp(Phi)), '-+k')
plt.xlabel("Temps adimensionné t / (dc / Vp)")
plt.ylabel("log10(V / Vp)")
plt.title("Cycles sismiques (Modèle patin-ressort)")
plt.show()
plt.plot()
# Pour afficher en unités réelles (années et m/s), décommenter ci-dessous :
T_annees = T * (pd.dc / pd.Vp) / (365.25 * 86400)
V_reel = np.exp(Phi) * pd.Vp
plt.plot(T_annees, V_reel, '-+k')
plt.xlabel("Temps (années)")
plt.ylabel("Vitesse en m/s")
plt.show()







