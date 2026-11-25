
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. GRILLE ET PARAMÈTRES ADIMENSIONNELS
# ---------------------------------------------------------------------
n = 128
dx = 0.2
x = (np.arange(n) - n / 2) * dx
l = n * dx

# Paramètres mécaniques originaux 
alpha_n0 = 0.2       # facteur de contrainte normale effective de référence 
eta = 1.0e-8
a = np.ones(n)
b = np.ones(n)
tau_inf = np.zeros(n)
h = 10.0 * n * dx

# Paramètres de frottement dépendant de la vitesse et de l’état
v_ref = 1.0
theta0 = 1.0
mu0 = 0.6            # coefficient de friction de référence
biot = 1.0           # coefficient de Biot (distinct de alpha_n0)
dc = 1.0             # distance caractéristique de glissement, adimensionnelle

# Paramètres hydrauliques : valeurs adimensionnelles provisoires
d_h = 0.05           # diffusivité hydraulique
q_inj = 0.002        # amplitude de la source (pression / temps)
x_inj = 0.0          # position de l’injection
w_inj = 0.5          # largeur de la source
t_ramp = 0.05        # temps caractéristique de montée de la source

# Paramètres numériques : 
t_end = 1.0
h_initial = 0.001
tol = 1.0e-7
nitrkmax = 30
nitmax = 100000
hmin = 1.0e-12
hmax = 1.0e-2
safe = 0.8

# Conditions initiales :
v0 = np.ones(n) * 1.0
v0[np.abs(x) < 5.0] = 0.001
phi0 = np.log(v0)
nu0 = np.log(np.ones(n) * theta0)
p0 = np.zeros(n)             # surpression de fluide, adimensionnelle
u0 = np.zeros(n)             # glissement cumulé, adimensionnel

# ---------------------------------------------------------------------
# 2. OPÉRATEUR MÉCANIQUE NON LOCAL : noyau spectral original
# ---------------------------------------------------------------------
def gth(phi, h, n=n, dx=dx):
    """Calcule g_th."""
    velocity = np.exp(phi)
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    abs_k = np.abs(wave_number)

    kernel = np.zeros(n)
    nonzero = abs_k > 0.0
    z = np.exp(-4.0 * h * abs_k[nonzero])

    kernel[nonzero] = abs_k[nonzero] * (1.0 + z) / (1.0 - z)

    return np.fft.ifft(kernel * np.fft.fft(velocity)).real


# ---------------------------------------------------------------------
# 3. MODÈLE HYDRAULIQUE : diffusion 1D le long de la faille
# ---------------------------------------------------------------------
def injection_source(t):
    """Source gaussienne adimensionnelle avec une montée progressive."""
    ramp = 1.0 - np.exp(-max(t, 0.0) / t_ramp)
    shape = np.exp(-0.5 * ((x - x_inj) / w_inj) ** 2)
    return q_inj * ramp * shape


def laplacian_neumann(p):
    """Dérivée seconde avec des conditions de flux nul aux limites."""
    lap = np.zeros_like(p)
    lap[1:-1] = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / dx**2
    lap[0] = 2.0 * (p[1] - p[0]) / dx**2
    lap[-1] = 2.0 * (p[-2] - p[-1]) / dx**2
    return lap


def pressure_rate(t, p):
    """Équation de diffusion ; le stockage est inclus dans d_h et q_inj."""
    return d_h * laplacian_neumann(p) + injection_source(t)


# ---------------------------------------------------------------------
# 4. COUPLAGE FROTTEMENT-ÉTAT ET CONTRAINTE EFFECTIVE
# ---------------------------------------------------------------------
def rhs(t, y):
    """
    Vecteur d’état y = [phi(n), nu(n), p(n), u(n)].

    Le taux de contrainte mécanique est repris du code 1D original :
      -0.5*g_th - (v-1)/h + tau_inf

    Pour retrouver la structure algébrique du code fourni lorsque p=0,
    on adopte la loi constitutive de prototype suivante :
      tau_f = sigma_eff * [mu0 + a*ln(v/v_ref)]
              + b*ln(theta*v_ref/dc) + eta*v
      sigma_eff = alpha_n0 - biot*p

    En dérivant tau_f et en posant tau_dot_mech = tau_dot_f, on obtient
      phi_dot = [tau_dot_mech - b*nu_dot
                 - (mu0+a*ln(v/v_ref))*sigma_eff_dot]
                / [sigma_eff*a + eta*v]
    avec sigma_eff_dot = -biot*p_dot.

    À pression constante, avec v_ref=dc=1 et sigma_eff=alpha_n0, cette
    expression retrouve le numérateur et le dénominateur du code original.
    C’est une hypothèse de modélisation à vérifier dans la dérivation d’origine.
    """
    phi = y[0:n]
    nu = y[n:2*n]
    p = y[2*n:3*n]
    # u = y[3*n:4*n]

    v = np.exp(phi)
    theta = np.exp(nu)

    p_dot = pressure_rate(t, p)
    sigma_eff = alpha_n0 - biot * p
    if np.min(sigma_eff) <= 0.0:
        # La contrainte normale effective s’annule : cette formulation
        # n’est plus valable au-delà de cet instant.
        raise RuntimeError(
            "La contrainte normale effective est nulle ou négative : réduisez l’injection ou révisez le modèle."
        )

    g_th = gth(phi, h, n, dx)
    tau_dot_mech = -0.5 * g_th - (v - 1.0) / h + tau_inf

    # Loi d’évolution de l’état, cohérente avec le code original si dc=v_ref=1.
    nu_dot = (v_ref / dc) * np.exp(-nu) - v / dc

    # Terme de frottement direct. Le terme b*ln(theta) reste
    # non multiplié par sigma_eff pour retrouver le code fourni lorsque p=0.
    mu_direct = mu0 + a * np.log(v / v_ref)

    sigma_eff_dot = -biot * p_dot
    denominator = sigma_eff * a + eta * v

    phi_dot = (
        tau_dot_mech
        - b * nu_dot
        - mu_direct * sigma_eff_dot
    ) / denominator

    u_dot = v

    return np.concatenate([phi_dot, nu_dot, p_dot, u_dot])


# ---------------------------------------------------------------------
# 5. INTÉGRATION TEMPORELLE RKF45 (FEHLBERG), STOCKAGE
# ---------------------------------------------------------------------
# Les coefficients et l'estimation d'erreur reprennent ceux des deux
# programmes d'origine. Ici, le même schéma est appliqué au vecteur complet
# [phi, nu, pression, glissement]. Le temps avance avec le pas réellement
# accepté, et non avec le pas recalculé pour l'itération suivante.

def rkf45_pas(t, y, pas):
    c21 = 1/4
    c31, c32 = 3/32, 9/32
    c41, c42, c43 = 1932/2197, -7200/2197, 7296/2197
    c51, c52, c53, c54 = 439/216, -8, 3680/513, -845/4104
    c61, c62, c63, c64, c65 = -8/27, 2, -3544/2565, 1859/4104, -11/40

    r1, r3, r4, r5, r6 = 1/360, -128/4275, -2197/75240, 1/50, 2/55
    c1, c3, c4, c5, c6 = 25/216, 1408/2565, 2197/4104, -1/5, 2/55

    for _ in range(nitrkmax):
        k1 = pas * rhs(t, y)
        k2 = pas * rhs(t + pas/4, y + c21*k1)
        k3 = pas * rhs(t + 3*pas/8, y + c31*k1 + c32*k2)
        k4 = pas * rhs(t + 12*pas/13, y + c41*k1 + c42*k2 + c43*k3)
        k5 = pas * rhs(t + pas, y + c51*k1 + c52*k2 + c53*k3 + c54*k4)
        k6 = pas * rhs(t + pas/2, y + c61*k1 + c62*k2 + c63*k3 + c64*k4 + c65*k5)

        erreur = r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6
        erreur_max = float(np.max(np.abs(erreur)))

        if np.isfinite(erreur_max) and erreur_max <= tol:
            y_nouveau = y + c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6
            if erreur_max > 0.0:
                pas_suivant = min(safe*pas*(tol/erreur_max)**0.25, hmax)
            else:
                pas_suivant = hmax
            return y_nouveau, pas_suivant, pas

        if not np.isfinite(erreur_max):
            pas = max(pas/2.0, hmin)
        else:
            pas = max(safe*pas*(tol/erreur_max)**0.25, hmin)

        if pas <= hmin and erreur_max > tol:
            raise RuntimeError("RKF45 : tolérance impossible à atteindre au pas minimal.")

    raise RuntimeError("RKF45 : nombre maximal de réessais dépassé pour un pas.")


y = np.concatenate([phi0, nu0, p0, u0])
t = 0.0
pas = h_initial

T = [t]
Y = [y.copy()]

for iteration in range(nitmax):
    if t >= t_end:
        break

    # Ne pas dépasser la durée finale avec le pas courant.
    pas = min(pas, t_end - t)
    y, pas_suivant, pas_utilise = rkf45_pas(t, y, pas)
    t += pas_utilise
    pas = pas_suivant

    # Stockage à chaque pas accepté : aucune phase rapide n'est écartée.
    T.append(t)
    Y.append(y.copy())

    if iteration % 100 == 0:
        phi_courant = y[0:n]
        print(
            f"itération {iteration} / {nitmax} ; "
            f"temps = {t:.6g} ; "
            f"vitesse maximale = {np.exp(np.max(phi_courant)):.6g} ; "
            f"pas suivant = {pas:.3g}"
        )
else:
    raise RuntimeError("RKF45 : nombre maximal d'itérations global atteint avant t_end.")

T = np.asarray(T)
Y = np.asarray(Y)
phi = Y[:, 0:n].T
nu = Y[:, n:2*n].T
pressure = Y[:, 2*n:3*n].T
slip = Y[:, 3*n:4*n].T

velocity = np.exp(phi)
theta = np.exp(nu)
sigma_eff = alpha_n0 - biot * pressure

# -----------------------------------------------------------------
# 6. DIAGNOSTICS ET GRAPHIQUES
# -----------------------------------------------------------------


# Même grandeur que le graphique du code 1D original : log10(V_max).
fig, ax = plt.subplots()
ax.plot(T, np.log10(np.max(velocity, axis=0)), "-+k", label=r"$\log_{10}(V_{max})$")
ax.set_xlabel("Temps adimensionnel")
ax.set_ylabel(r"$\log_{10}(V_{max})$")
ax.grid(True)
ax.legend()

fig, ax = plt.subplots()
log_velocity = np.log10(np.maximum(velocity.T, 1e-30))
# Fixer explicitement l'échelle évite une carte uniforme ou illisible.
im = ax.pcolormesh(
    x, T, log_velocity,
    shading="auto", cmap="viridis",
    vmin=float(np.nanmin(log_velocity)),
    vmax=float(np.nanmax(log_velocity))
)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(T[0], T[-1])
ax.set_xlabel("Position le long de la faille x (adimensionnelle)")
ax.set_ylabel("Temps adimensionnel")
ax.set_title(r"$\log_{10}(v(x,t))$")
fig.colorbar(im, ax=ax, label=r"$\log_{10}(v)$")

fig, ax = plt.subplots()
im = ax.pcolormesh(x, T, pressure.T, shading="auto")
ax.set_xlabel("Position le long de la faille x (adimensionnelle)")
ax.set_ylabel("Temps adimensionnel")
ax.set_title("Surpression de fluide p(x,t)")
fig.colorbar(im, ax=ax, label="Pression adimensionnelle")

fig, ax = plt.subplots()
log_theta = np.log10(np.maximum(theta.T, 1e-30))
im = ax.pcolormesh(
    x, T, log_theta,
    shading="auto", cmap="plasma",
    vmin=float(np.nanmin(log_theta)),
    vmax=float(np.nanmax(log_theta))
)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(T[0], T[-1])
ax.set_xlabel("Position le long de la faille x (adimensionnelle)")
ax.set_ylabel("Temps adimensionnel")
ax.set_title(r"$\log_{10}(\theta(x,t))$")
fig.colorbar(im, ax=ax, label=r"$\log_{10}(\theta)$")

# Évolution de la vitesse en plusieurs positions représentatives.
fig, ax = plt.subplots()
for x_target in (x_inj, -5.0, 5.0, 10.0):
    idx = int(np.argmin(np.abs(x - x_target)))
    ax.plot(T, velocity[idx, :], label=f"x={x[idx]:.2f}")
ax.set_xlabel("Temps adimensionnel")
ax.set_ylabel("Vitesse de glissement v")
ax.set_yscale("log")
ax.grid(True)
ax.legend()

# Profils du glissement cumulé à plusieurs instants.
fig, ax = plt.subplots()
for j in (0, len(T)//3, 2*len(T)//3, -1):
    ax.plot(x, slip[:, j], label=f"t={T[j]:.3g}")
ax.set_xlabel("Position le long de la faille x")
ax.set_ylabel("Glissement cumulé u")
ax.grid(True)
ax.legend()

# Contrainte normale effective.
fig, ax = plt.subplots()
im = ax.pcolormesh(x, T, sigma_eff.T, shading="auto")
ax.set_xlabel("Position le long de la faille x")
ax.set_ylabel("Temps adimensionnel")
ax.set_title("Contrainte normale effective")
fig.colorbar(im, ax=ax, label=r"$\sigma'_n$ (adimensionnelle)")

plt.show()

