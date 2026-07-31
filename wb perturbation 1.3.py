import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# CONFIG
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_script_dir, "results")

os.makedirs(output_dir, exist_ok=True)

print(f"Le GIF sera sauvegardé ici : {output_dir}")
g = 9.81
Nx = 100 
Tfinal = 2.0
CFL = 0.5
xL, xR = 0.0, 1.0
dx = (xR - xL) / Nx
x = xL + (np.arange(Nx) + 0.5) * dx

def get_equilibrium_state(x_grid):
    """Retourne l'état d'équilibre avec la topographie complexe. """
    # 1. Topographie complexe : Bosse + Sinusoïde
    z = 0.3 * np.exp(-100 * (x_grid - 0.5)**2) + 0.1 * np.sin(4 * np.pi * x_grid)
    
    # 2. Paramètres de hauteur et débit
    h0_val = 1.0
    h = h0_val * np.ones_like(x_grid)
    q = np.zeros_like(x_grid)
    
    # 3. Calcul de m via theta (calcul interne)
    # On calcule theta pour avoir m, mais on ne le retourne pas à la fin
    theta = np.exp(-2.0 * z / h0_val)
    m = h * theta 
    
    # 4. Retourne UNIQUEMENT les 5 valeurs attendues par votre script
    # Correspond à : h_eq_glob, q_eq_glob, m_eq_glob, z_glob, h0_glob
    return h, q, m, z, h0_val

def get_equilibrium_state(x_grid):
    """ Cas 3 : h = h0 (cste), u = 0, theta variable.
    Équilibre théorique : z + (h0/2)*ln(theta) = C """
    A = 0.05
    h0_scalar = 1.0
    C = 0.0
    
    # 1. Topographie sinus
    z = A * np.sin(2.0 * np.pi * x_grid)
    
    # 2. Hauteur constante
    h = h0_scalar * np.ones_like(x_grid)
    
    # 3. Vitesse nulle
    u = np.zeros_like(x_grid)
    q = h * u
    
    # 4. Température dérivée pour satisfaire l'équilibre
    theta = np.exp((2.0 * (C - z)) / h0_scalar)
    m = h * theta
    
    # Données de référence pour les plots
    ref_invariant = C
    h0_ref = h0_scalar
    theta_ref = theta.copy()
    
    return h, q, m, z, h0_ref


# On stocke l'équilibre globalement pour y accéder dans le flux
h_eq_glob, q_eq_glob, m_eq_glob, z_glob, h0_glob = get_equilibrium_state(x)

def flux_physique_ripa(h, q, m):
    # Flux total F(U)
    eps = 1e-12
    h = np.maximum(h, eps)
    u = q / h
    p = 0.5 * g * m * h 
    F1 = q
    F2 = q * u + p
    F3 = m * u
    return F1, F2, F3

def reconstruction_weno5(u):
    """ Reconstruction WENO5 classique """
    u_m2 = np.roll(u, 2); u_m1 = np.roll(u, 1); u_0 = u
    u_p1 = np.roll(u, -1); u_p2 = np.roll(u, -2)
    
    # Gauche (Left)
    p0 = (2*u_m2 - 7*u_m1 + 11*u_0)/6
    p1 = (-u_m1  + 5*u_0  + 2*u_p1)/6
    p2 = (2*u_0  + 5*u_p1 - u_p2)/6
    b0 = (13/12)*(u_m2-2*u_m1+u_0)**2 + 0.25*(u_m2-4*u_m1+3*u_0)**2
    b1 = (13/12)*(u_m1-2*u_0+u_p1)**2 + 0.25*(u_m1-u_p1)**2
    b2 = (13/12)*(u_0-2*u_p1+u_p2)**2 + 0.25*(3*u_0-4*u_p1+u_p2)**2
    eps_w = 1e-6
    a0 = 0.1/(eps_w+b0)**2; a1 = 0.6/(eps_w+b1)**2; a2 = 0.3/(eps_w+b2)**2
    uL = (a0*p0 + a1*p1 + a2*p2) / (a0 + a1 + a2)

    # Droite (Right) - Miroir
    p0 = (2*u_p2 - 7*u_p1 + 11*u_0)/6
    p1 = (-u_p1  + 5*u_0  + 2*u_m1)/6
    p2 = (2*u_0  + 5*u_m1 - u_m2)/6
    b0 = (13/12)*(u_p2-2*u_p1+u_0)**2 + 0.25*(u_p2-4*u_p1+3*u_0)**2
    b1 = (13/12)*(u_p1-2*u_0+u_m1)**2 + 0.25*(u_p1-u_m1)**2
    b2 = (13/12)*(u_0-2*u_m1+u_m2)**2 + 0.25*(3*u_0-4*u_m1+u_m2)**2
    a0 = 0.1/(eps_w+b0)**2; a1 = 0.6/(eps_w+b1)**2; a2 = 0.3/(eps_w+b2)**2
    uR_at_i = (a0*p0 + a1*p1 + a2*p2) / (a0 + a1 + a2)
    
    return uL, np.roll(uR_at_i, -1)


# 3. CALCUL DU PAS DE TEMPS (CFL)
def compute_dt_CFL(h, q, m):
    """ Calcule le pas de temps stable selon la condition CFL """
    eps = 1e-12
    h = np.maximum(h, eps)
    u = q / h
    c = np.sqrt(g * m / h)
    
    # Vitesse d'onde max
    lambda_max = np.max(np.abs(u) + c)
    
    # dt = CFL * dx / max_speed
    dt = CFL * dx / (lambda_max + eps)
    
    # On borne dt pour éviter des pas trop grands si l'eau est calme, mais pas trop petits
    return min(dt, 0.01)

# 4. SOLVEUR DE DÉVIATION (HLLC sur Delta U)
import numpy as np

def flux_physique_ripa(h, q, m, g=9.81):
    """
    Calcule le flux physique F(U) pour le système de Ripa.
    F(U) = [ q,
             q^2/h + 0.5 * g * m * h,
             m * q / h ]
    """
    eps = 1e-12
    h = np.maximum(h, eps)
    u = q / h

    pression = 0.5 * g * m * h 
    
    f1 = q
    f2 = (q * u) + pression
    f3 = m * u
    return f1, f2, f3

def compute_hllc_flux_vectorized(hL, qL, mL, hR, qR, mR, g=9.81):
    """Calcul vectorisé du flux numérique HLLC.
    Sépare clairement le calcul des ondes, des états étoilés U* et des flux F*."""
    eps = 1e-12
    
    # 1. Variables Primitives & Célérités
    # u = q/h, theta = m/h
    uL = qL / np.maximum(hL, eps)
    uR = qR / np.maximum(hR, eps)
    
    # Célérité c = sqrt(g * theta * h) = sqrt(g * m) dans Ripa
    cL = np.sqrt(np.maximum(0, g * mL))
    cR = np.sqrt(np.maximum(0, g * mR))

    # 2. Vitesses d'ondes
    # S_L = min(u_L - c_L, u_R - c_R)
    # S_R = max(u_L + c_L, u_R + c_R)
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)

    # 3. Vitesse de l'onde de Contact S_*
    # Formule exacte pour restaurer la pression constante dans la zone étoilée
    pL = 0.5 * g * mL * hL
    pR = 0.5 * g * mR * hR
    
    num = pR - pL + qL * (SL - uL) - qR * (SR - uR)
    den = hL * (SL - uL) - hR * (SR - uR)
    S_star = num / (den + 1e-14)

    # 4. Calcul des États Étoilés U* (Rankine-Hugoniot)
    # U*_K = h_K * ((S_K - u_K) / (S_K - S_*)) * [1, S_*, m/h + ...]
    # Facteur de compression phi_K
    phi_L = hL * (SL - uL) / (SL - S_star + eps)
    phi_R = hR * (SR - uR) / (SR - S_star + eps)

    # U*_L (État intermédiaire gauche)
    hL_star = phi_L
    qL_star = phi_L * S_star
    mL_star = phi_L * (mL / np.maximum(hL, eps)) # La température est advectée
    
    # U*_R (État intermédiaire droite)
    hR_star = phi_R
    qR_star = phi_R * S_star
    mR_star = phi_R * (mR / np.maximum(hR, eps))

    # 5. Calcul des Flux HLLC par zones
    # Flux physiques de base
    F1L, F2L, F3L = flux_physique_ripa(hL, qL, mL, g)
    F1R, F2R, F3R = flux_physique_ripa(hR, qR, mR, g)
    
    # Initialisation des flux numériques
    F1_num = np.zeros_like(hL)
    F2_num = np.zeros_like(hL)
    F3_num = np.zeros_like(hL)

    # Masques logiques pour les 4 régions
    mask_L = (SL >= 0)
    
    # Formule HLLC : F*_L = F_L + S_L * (U*_L - U_L)
    mask_sL = (SL < 0) & (S_star >= 0)
    
    # Formule HLLC : F*_R = F_R + S_R * (U*_R - U_R)
    mask_sR = (S_star < 0) & (SR > 0)
    
    mask_R = (SR <= 0)

    # Remplissage : Cas simple (F_L ou F_R)
    F1_num[mask_L], F2_num[mask_L], F3_num[mask_L] = F1L[mask_L], F2L[mask_L], F3L[mask_L]
    F1_num[mask_R], F2_num[mask_R], F3_num[mask_R] = F1R[mask_R], F2R[mask_R], F3R[mask_R]

    # Remplissage : Cas étoilé (F*_L)
    # dF = S_L * (U* - U)
    F1_num[mask_sL] = F1L[mask_sL] + SL[mask_sL] * (hL_star[mask_sL] - hL[mask_sL])
    F2_num[mask_sL] = F2L[mask_sL] + SL[mask_sL] * (qL_star[mask_sL] - qL[mask_sL])
    F3_num[mask_sL] = F3L[mask_sL] + SL[mask_sL] * (mL_star[mask_sL] - mL[mask_sL])

    # Remplissage : Cas étoilé (F*_R)
    # dF = S_R * (U* - U)
    F1_num[mask_sR] = F1R[mask_sR] + SR[mask_sR] * (hR_star[mask_sR] - hR[mask_sR])
    F2_num[mask_sR] = F2R[mask_sR] + SR[mask_sR] * (qR_star[mask_sR] - qR[mask_sR])
    F3_num[mask_sR] = F3R[mask_sR] + SR[mask_sR] * (mR_star[mask_sR] - mR[mask_sR])

    return F1_num, F2_num, F3_num

def integrate_source_simpson(hL, mL, hR, mR, g=9.81):
    """Intégration du terme source S = -g * h * theta * dz/dx via la méthode de Simpson (1/6, 4/6, 1/6) pour préserver l'équilibre.
    Retourne la partie intégrale."""
    eps = 1e-12
    
    # État Gauche
    thetaL = mL / np.maximum(hL, eps)
    S_L = -g * hL * thetaL
    
    # État Droite
    thetaR = mR / np.maximum(hR, eps)
    S_R = -g * hR * thetaR
    
    # État Milieu
    h_mid = 0.5 * (hL + hR)
    m_mid = 0.5 * (mL + mR)
    theta_mid = m_mid / np.maximum(h_mid, eps)
    S_mid = -g * h_mid * theta_mid
    
    # Intégrale de Simpson : (S_L + 4*S_mid + S_R) / 6
    return (S_L + 4 * S_mid + S_R) / 6.0

def solveur_deviation(du_h, du_q, du_m):
    """Résout dU/dt + div(F_tot - F_eq) = S_tot - S_eq. J'utilise HLLC pour F_tot ET F_eq pour garantir la consistance.
    """
    eps = 1e-12
    
    # 1. Reconstruction WENO5
    # Perturbations
    dhL, dhR = reconstruction_weno5(du_h)
    dqL, dqR = reconstruction_weno5(du_q)
    dmL, dmR = reconstruction_weno5(du_m)
    
    # Équilibre (variables supposées globales ou passées en arg)
    h_eqL, h_eqR = reconstruction_weno5(h_eq_glob)
    q_eqL, q_eqR = reconstruction_weno5(q_eq_glob)
    m_eqL, m_eqR = reconstruction_weno5(m_eq_glob)
    zL, zR = reconstruction_weno5(z_glob)
    
    # 2. Construction des États Totaux aux interfaces
    # U_tot = U_eq + dU
    hL_tot = np.maximum(h_eqL + dhL, eps); hR_tot = np.maximum(h_eqR + dhR, eps)
    qL_tot = q_eqL + dqL;                  qR_tot = q_eqR + dqR
    mL_tot = m_eqL + dmL;                  mR_tot = m_eqR + dmR
    
    # 3. Flux Numérique Total (HLLC sur U_tot)
    F1_num, F2_num, F3_num = compute_hllc_flux_vectorized(
        hL_tot, qL_tot, mL_tot, 
        hR_tot, qR_tot, mR_tot
    )
    
    # 4. Flux Numérique à l'Équilibre (HLLC sur U_eq)
    F1_eq, F2_eq, F3_eq = compute_hllc_flux_vectorized(
        h_eqL, q_eqL, m_eqL, 
        h_eqR, q_eqR, m_eqR
    )
    
    # 5. Écart de Flux (Net Flux)
    dF1 = F1_num - F1_eq
    dF2 = F2_num - F2_eq
    dF3 = F3_num - F3_eq
    
    # 6. Terme Source Well-Balanced (Simpson)
    # On calcule l'intégrale de S sur l'état total et l'état d'équilibre
    Src_integral_tot = integrate_source_simpson(hL_tot, mL_tot, hR_tot, mR_tot)
    Src_integral_eq  = integrate_source_simpson(h_eqL, m_eqL, h_eqR, m_eqR)
    
    # Le terme source net est la différence * saut de topographie
    # S_net = (Int(S_tot) - Int(S_eq)) * Delta_Z
    dS_interface = (Src_integral_tot - Src_integral_eq) * (zR - zL)
    
    # Ajout au flux de quantité de mouvement (Eq 27 du rapport : F* - 1/2 S*)
    dF2 -= 0.5 * dS_interface
    
    return dF1, dF2, dF3

def compute_rhs_deviation(du_h, du_q, du_m):
    dF1, dF2, dF3 = solveur_deviation(du_h, du_q, du_m)
    
    rhs_h = - (dF1 - np.roll(dF1, 1)) / dx
    rhs_q = - (dF2 - np.roll(dF2, 1)) / dx
    rhs_m = - (dF3 - np.roll(dF3, 1)) / dx
    
    # Terme Source Volume Net
    dz = (np.roll(z_glob, -1) - np.roll(z_glob, 1)) / (2*dx)
    
    h_tot = h_eq_glob + du_h
    m_tot = m_eq_glob + du_m
    theta_tot = m_tot / np.maximum(h_tot, 1e-12)
    S_vol_tot = -g * h_tot * theta_tot * dz
    
    theta_eq = m_eq_glob / np.maximum(h_eq_glob, 1e-12)
    S_vol_eq = -g * h_eq_glob * theta_eq * dz
    
    rhs_q += (S_vol_tot - S_vol_eq)
    
    return rhs_h, rhs_q, rhs_m

# 5. SIMULATION
def simulate():
    du_h = np.zeros_like(x)
    du_q = np.zeros_like(x)
    du_m = np.zeros_like(x)
    
    times = [0.0]
    H_hist = [(h_eq_glob + du_h + z_glob).copy()]
    
    h_tot = h_eq_glob + du_h
    m_tot = m_eq_glob + du_m
    theta_tot = m_tot / np.maximum(h_tot, 1e-12)
    inv = z_glob + (h0_glob/2.0)*np.log(np.maximum(theta_tot, 1e-12))
    
    inv_hist = [inv.copy()]
    th_hist = [theta_tot.copy()]
    
    t = 0.0
    next_save = 0.05
    
    print("Début simulation (Méthode de Déviation - Perturbation)...")
    
    while t < Tfinal:
        h_tot = h_eq_glob + du_h
        q_tot = q_eq_glob + du_q
        m_tot = m_eq_glob + du_m
        dt = compute_dt_CFL(h_tot, q_tot, m_tot)
        if t + dt > Tfinal: dt = Tfinal - t
        
        # RK4
        k1h, k1q, k1m = compute_rhs_deviation(du_h, du_q, du_m)
        
        du_h2 = du_h + 0.5*dt*k1h; du_q2 = du_q + 0.5*dt*k1q; du_m2 = du_m + 0.5*dt*k1m
        k2h, k2q, k2m = compute_rhs_deviation(du_h2, du_q2, du_m2)
        
        du_h3 = du_h + 0.5*dt*k2h; du_q3 = du_q + 0.5*dt*k2q; du_m3 = du_m + 0.5*dt*k2m
        k3h, k3q, k3m = compute_rhs_deviation(du_h3, du_q3, du_m3)
        
        du_h4 = du_h + dt*k3h; du_q4 = du_q + dt*k3q; du_m4 = du_m + dt*k3m
        k4h, k4q, k4m = compute_rhs_deviation(du_h4, du_q4, du_m4)
        
        du_h += (dt/6.0)*(k1h + 2*k2h + 2*k3h + k4h)
        du_q += (dt/6.0)*(k1q + 2*k2q + 2*k3q + k4q)
        du_m += (dt/6.0)*(k1m + 2*k2m + 2*k3m + k4m)
        
        t += dt
        
        if t >= next_save:
            times.append(t)
            h_tot = h_eq_glob + du_h
            m_tot = m_eq_glob + du_m
            theta_tot = m_tot / np.maximum(h_tot, 1e-12)
            curr_inv = z_glob + (h0_glob / 2.0) * np.log(np.maximum(theta_tot, 1e-12))
            
            H_hist.append((h_tot+z_glob).copy())
            th_hist.append(theta_tot.copy())
            inv_hist.append(curr_inv.copy())
            next_save += 0.05
            print(f"t = {t:.3f}s (dt={dt:.1e})")
    
    inv_matrix = np.array(inv_hist)
    err = np.max(np.abs(inv_matrix))

    # Dans la méthode de déviation (Well-Balanced), l'état exact est l'équilibre.
    # Donc l'erreur est exactement la valeur de la perturbation (du).
    err_h = np.max(np.abs(du_h))
    err_q = np.max(np.abs(du_q))
    err_m = np.max(np.abs(du_m))

    print("-" * 50)
    print(f"ERREUR MAX (Deviation Form) sur l'invariant : {err:.5e}")
    print(f"Erreur max h : {err_h:.2e}")
    print(f"Erreur max q : {err_q:.2e}")
    print(f"Erreur max m : {err_m:.2e}")
    print("-" * 50)
    
    return times, H_hist, th_hist, inv_hist, z_glob, h0_glob

# 6. ANIMATION
def create_gif(times, H_hist, th_hist, inv_hist, z, h0_val, x_grid):
    print("Création du GIF...")
    if len(times) == 0: return

    H_arr = np.array(H_hist)
    th_arr = np.array(th_hist)
    inv_arr = np.array(inv_hist)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Modèle de Ripa 1D — Well Balanced\n"
                 r"Cas 3: $z + \frac{h_0}{2}\ln(\theta) = cst$", fontsize=14)
    plt.subplots_adjust(hspace=0.4, top=0.90, left=0.15)
    ax1, ax2, ax3 = axs
    
        # Graph 1 : Surface
    y_min = np.min(z) - 0.2
    y_max = np.max(H_arr) + 0.2
    ax1.fill_between(x_grid, -2, z, color="saddlebrown", alpha=0.5, label="Fond")
    ax1.plot(x_grid, z, "k-", lw=1)
    water_poly = [ax1.fill_between(x_grid, z, H_arr[0], color="deepskyblue", alpha=0.5, label="Eau")]
    lineH, = ax1.plot(x_grid, H_arr[0], "b-", lw=2, label="Surface")
    ax1.set_ylabel(r"$h + z$")
#    ax1.set_ylim(y_min, y_max)
    ax1.set_ylim(-0.5, 1.7)
    ax1.legend(loc="upper right", fontsize=9)
    time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes, fontsize=12,
                         bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    
    # Graph 2 : Erreur Invariant
    ax2.axhline(0, color='r', linestyle='--', label="Théorique = 0")
    lineInv, = ax2.plot(x_grid, inv_arr[0], "g-", label="Calculé")
    ax2.set_ylim(-1e-7, 1e-7)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.set_ylabel(r"$z + \frac{h_0}{2}\ln(\theta)$")
    ax2.legend(loc="upper right", fontsize=9)
    
    # Graph 3 : Température
    th_theo = np.exp(-2.0*z/h0_val)
    lineTh, = ax3.plot(x_grid, th_arr[0], "r-", lw=2, label=r"$\theta$ Calculé")
    ax3.plot(x_grid, th_theo, "k--", label=r"$\theta$ Théorique")
    ax3.set_ylabel(r"$\theta$")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=9)
    
    def update(i):
        lineH.set_data(x_grid, H_arr[i])
        lineInv.set_data(x_grid, inv_arr[i])
        lineTh.set_data(x_grid, th_arr[i])
        water_poly[0].remove()
        water_poly[0] = ax1.fill_between(x_grid, z, H_arr[i], color="deepskyblue", alpha=0.5)
        time_text.set_text(f"t = {times[i]:.3f} s")
        return lineH, lineInv, lineTh, water_poly[0], time_text

    anim = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
    gif_path = os.path.join(output_dir, "Cas 3 wb pertu.gif")
    
    try:
        print(f"Sauvegarde en cours vers {gif_path} ...")
        anim.save(gif_path, writer=PillowWriter(fps=15))
        print("Succès ! GIF sauvegardé.")
    except Exception as e:
        print(f"Erreur : {e}")
    
    print("Sauvegarde de l'image finale (PNG)...")
    
    last_frame_index = len(times) - 1
    update(last_frame_index)
    
    png_path = os.path.join(output_dir, "Cas 3 wb pertu.png")
    fig.savefig(png_path, dpi=200) 
    print(f"Succès ! Image sauvegardée : {png_path}")

if __name__ == "__main__":
    times, H_hist, th_hist, inv_hist, z, h0_val = simulate()
    create_gif(times, H_hist, th_hist, inv_hist, z, h0_val, x)