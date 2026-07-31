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

# Paramètres 
g = 9.81
Nx = 200 
Tfinal = 2.0
CFL = 0.5
xL, xR = 0.0, 1.0
dx = (xR - xL) / Nx
x = xL + (np.arange(Nx) + 0.5) * dx

# 1. INITIALISATION & EQUILIBRE EXACT (CAS 1)


def get_equilibrium_state_case1(x_grid):
    """
    Retourne l'état d'équilibre exact pour le CAS 1. Lac au repos : u=0, h+z=H0, theta=theta0.
    """
    # Paramètres
    A = 0.2
    H0 = 1.0
    theta0 = 1.0
    
    # 1. Topographie variable (sinusoïdale)
    z = A * np.sin(2.0 * np.pi * x_grid)
    
    # 2. Hauteur d'eau (surface libre plate H0)
    h_eq = np.maximum(0.0, H0 - z)
    
    # 3. Vitesse nulle
    q_eq = np.zeros_like(x_grid)
    
    # 4. Température constante
    m_eq = h_eq * theta0
    
    return h_eq, q_eq, m_eq, z, H0, theta0

# On stocke l'équilibre globalement
h_eq_glob, q_eq_glob, m_eq_glob, z_glob, H0_glob, theta0_glob = get_equilibrium_state_case1(x)

# 2. FLUX ET RECONSTRUCTION
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
    """ Reconstruction WENO5 classique avec gestion des bords périodiques """
    u_m2 = np.roll(u, 2); u_m1 = np.roll(u, 1); u_0 = u
    u_p1 = np.roll(u, -1); u_p2 = np.roll(u, -2)
    
    # Gauche (Left)
    beta0 = (13/12)*(u_m2 - 2*u_m1 + u_0)**2 + 0.25*(u_m2 - 4*u_m1 + 3*u_0)**2
    beta1 = (13/12)*(u_m1 - 2*u_0 + u_p1)**2 + 0.25*(u_m1 - u_p1)**2
    beta2 = (13/12)*(u_0 - 2*u_p1 + u_p2)**2 + 0.25*(3*u_0 - 4*u_p1 + u_p2)**2
    
    eps_w = 1e-6
    alpha0 = 0.1 / (eps_w + beta0)**2
    alpha1 = 0.6 / (eps_w + beta1)**2
    alpha2 = 0.3 / (eps_w + beta2)**2
    omega_sum = alpha0 + alpha1 + alpha2
    w0 = alpha0 / omega_sum; w1 = alpha1 / omega_sum; w2 = alpha2 / omega_sum
    
    p0 = (2*u_m2 - 7*u_m1 + 11*u_0)/6
    p1 = (-u_m1 + 5*u_0 + 2*u_p1)/6
    p2 = (2*u_0 + 5*u_p1 - u_p2)/6
    
    uL = w0*p0 + w1*p1 + w2*p2

    # Droite (Right) - Miroir
    beta0_r = (13/12)*(u_p2 - 2*u_p1 + u_0)**2 + 0.25*(u_p2 - 4*u_p1 + 3*u_0)**2
    beta1_r = (13/12)*(u_p1 - 2*u_0 + u_m1)**2 + 0.25*(u_p1 - u_m1)**2
    beta2_r = (13/12)*(u_0 - 2*u_m1 + u_m2)**2 + 0.25*(3*u_0 - 4*u_m1 + u_m2)**2
    
    alpha0_r = 0.1 / (eps_w + beta0_r)**2
    alpha1_r = 0.6 / (eps_w + beta1_r)**2
    alpha2_r = 0.3 / (eps_w + beta2_r)**2
    omega_sum_r = alpha0_r + alpha1_r + alpha2_r
    
    p0_r = (2*u_p2 - 7*u_p1 + 11*u_0)/6
    p1_r = (-u_p1 + 5*u_0 + 2*u_m1)/6
    p2_r = (2*u_0 + 5*u_m1 - u_m2)/6
    
    uR_at_i = (alpha0_r*p0_r + alpha1_r*p1_r + alpha2_r*p2_r) / omega_sum_r
    
    return uL, np.roll(uR_at_i, -1)


# 3. CALCUL DU PAS DE TEMPS (CFL)
def compute_dt_CFL(h, q, m):
    eps = 1e-12
    h = np.maximum(h, eps)
    u = q / h
    c = np.sqrt(g * m / h)
    lambda_max = np.max(np.abs(u) + c)
    dt = CFL * dx / (lambda_max + eps)
    return min(dt, 0.01)

# 4. SOLVEUR DE DÉVIATION (HLLC sur Delta U)
def compute_hllc_flux_vectorized(hL, qL, mL, hR, qR, mR, g=9.81):
    eps = 1e-12
    
    uL = qL / np.maximum(hL, eps); uR = qR / np.maximum(hR, eps)
    cL = np.sqrt(np.maximum(0, g * mL)); cR = np.sqrt(np.maximum(0, g * mR))

    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)

    pL = 0.5 * g * mL * hL; pR = 0.5 * g * mR * hR
    num = pR - pL + qL * (SL - uL) - qR * (SR - uR)
    den = hL * (SL - uL) - hR * (SR - uR)
    S_star = num / (den + 1e-14)

    phi_L = hL * (SL - uL) / (SL - S_star + eps)
    phi_R = hR * (SR - uR) / (SR - S_star + eps)

    hL_star = phi_L; qL_star = phi_L * S_star; mL_star = phi_L * (mL / np.maximum(hL, eps))
    hR_star = phi_R; qR_star = phi_R * S_star; mR_star = phi_R * (mR / np.maximum(hR, eps))

    F1L, F2L, F3L = flux_physique_ripa(hL, qL, mL)
    F1R, F2R, F3R = flux_physique_ripa(hR, qR, mR)
    
    F1_num = np.zeros_like(hL); F2_num = np.zeros_like(hL); F3_num = np.zeros_like(hL)

    mask_L = (SL >= 0)
    mask_sL = (SL < 0) & (S_star >= 0)
    mask_sR = (S_star < 0) & (SR > 0)
    mask_R = (SR <= 0)

    F1_num[mask_L], F2_num[mask_L], F3_num[mask_L] = F1L[mask_L], F2L[mask_L], F3L[mask_L]
    F1_num[mask_R], F2_num[mask_R], F3_num[mask_R] = F1R[mask_R], F2R[mask_R], F3R[mask_R]

    F1_num[mask_sL] = F1L[mask_sL] + SL[mask_sL] * (hL_star[mask_sL] - hL[mask_sL])
    F2_num[mask_sL] = F2L[mask_sL] + SL[mask_sL] * (qL_star[mask_sL] - qL[mask_sL])
    F3_num[mask_sL] = F3L[mask_sL] + SL[mask_sL] * (mL_star[mask_sL] - mL[mask_sL])

    F1_num[mask_sR] = F1R[mask_sR] + SR[mask_sR] * (hR_star[mask_sR] - hR[mask_sR])
    F2_num[mask_sR] = F2R[mask_sR] + SR[mask_sR] * (qR_star[mask_sR] - qR[mask_sR])
    F3_num[mask_sR] = F3R[mask_sR] + SR[mask_sR] * (mR_star[mask_sR] - mR[mask_sR])

    return F1_num, F2_num, F3_num

def integrate_source_simpson(hL, mL, hR, mR, g=9.81):
    """ Intégration du terme source -g*h*theta pour Simpson (sans le dz) """
    eps = 1e-12
    thetaL = mL / np.maximum(hL, eps)
    S_L = -g * hL * thetaL
    
    thetaR = mR / np.maximum(hR, eps)
    S_R = -g * hR * thetaR
    
    h_mid = 0.5 * (hL + hR)
    m_mid = 0.5 * (mL + mR)
    theta_mid = m_mid / np.maximum(h_mid, eps)
    S_mid = -g * h_mid * theta_mid
    
    return (S_L + 4 * S_mid + S_R) / 6.0

def solveur_deviation(du_h, du_q, du_m):
    """
    Solveur Well-Balanced par perturbation / Résout dU/dt + div(F_tot - F_eq) = S_tot - S_eq / Utilise HLLC pour F_tot ET F_eq pour garantir la consistance exacte.
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
    
    #  2. Construction des États Totaux aux interfaces 
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
    # Au lieu de faire la moyenne des flux physiques, on applique le meem solveur numérique (HLLC) aux états d'équilibre reconstruits.
    # Ainsi, si dU = 0 (donc U_tot == U_eq), alors F_num est identique à F_eq.
    F1_eq, F2_eq, F3_eq = compute_hllc_flux_vectorized(
        h_eqL, q_eqL, m_eqL, 
        h_eqR, q_eqR, m_eqR
    )
    
    # 5. Écart de Flux 
    
    dF1 = F1_num - F1_eq
    dF2 = F2_num - F2_eq
    dF3 = F3_num - F3_eq
    
    # 6. Terme Source Well-Balanced (Simpson) 
    # On calcule l'intégrale de S sur l'état total et l'état d'équilibre
    Src_integral_tot = integrate_source_simpson(hL_tot, mL_tot, hR_tot, mR_tot)
    
    # ici on passe bien les états d'équilibre gauche et droite (m_eqR)
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
    # Perturbations initiales = 0
    du_h = np.zeros_like(x)
    du_q = np.zeros_like(x)
    du_m = np.zeros_like(x)
    
    times = [0.0]
    
    # Init Histo
    h_tot = h_eq_glob + du_h
    m_tot = m_eq_glob + du_m
    theta_tot = m_tot / np.maximum(h_tot, 1e-12)
    
    H_hist = [(h_tot + z_glob).copy()]
    th_hist = [theta_tot.copy()]
    
    t = 0.0
    next_save = 0.05
    
    print("Début simulation Cas 1 (WB Perturbation)...")
    
    while t < Tfinal:
        # DT
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
            
            H_hist.append((h_tot+z_glob).copy())
            th_hist.append(theta_tot.copy())
            next_save += 0.05
            print(f"t = {t:.3f}s (dt={dt:.1e})")
    
    # Erreur finale sur H 
    H_final = H_hist[-1]
    err_H = np.max(np.abs(H_final - H0_glob))
    
    # Erreurs max sur les perturbations (doivent être ~0)
    err_h = np.max(np.abs(du_h))
    err_q = np.max(np.abs(du_q))
    err_m = np.max(np.abs(du_m))

    print("-" * 50)
    print(f"ERREUR MAX Invariant H : {err_H:.5e}")
    print(f"Erreur max Perturbation h : {err_h:.2e}")
    print(f"Erreur max Perturbation q : {err_q:.2e}")
    print(f"Erreur max Perturbation m : {err_m:.2e}")
    print("-" * 50)
    
    return times, H_hist, th_hist, z_glob, H0_glob, theta0_glob

# 6. ANIMATION
def create_gif(times, H_hist, th_hist, z, H0_val, theta0_val, x_grid):
    print("Création du GIF...")
    if len(times) == 0: return

    H_arr = np.array(H_hist)
    th_arr = np.array(th_hist)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Modèle de Ripa 1D — Well Balanced (Perturbation)\n"
                 r"Cas 1: Lac au repos, $h+z = H_0, \theta=\theta_0$", fontsize=14)
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
    ax1.set_ylim(-0.5, 1.7)
    ax1.legend(loc="upper right", fontsize=9)
    time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes, fontsize=12,
                         bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    
    # Graph 2 : Invariant H 
    ax2.plot(x_grid, H0_val * np.ones_like(x_grid), "r--", label=f"Exact ($H_0={H0_val}$)")
    lineInv, = ax2.plot(x_grid, H_arr[0], "k-", label="Calculé")

    margin = 1e-10
    ax2.set_ylim(H0_val - 1e-9, H0_val + 1e-9) 
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.set_ylabel(r"Surface $H = h+z$")
    ax2.legend(loc="upper right", fontsize=9)
    
    # Graph 3 : Température
    ax3.plot(x_grid, theta0_val * np.ones_like(x_grid), "r--", label=f"Exact ($\t_0={theta0_val}$)")
    lineTh, = ax3.plot(x_grid, th_arr[0], "k-", label="Calculé")
    ax3.set_ylabel(r"Température $\theta$")
    ax3.grid(True, alpha=0.3)
    
    # Ajustement d'échelle
    th_min, th_max = theta0_val, theta0_val
    pad = 0.1
    ax3.set_ylim(th_min - pad, th_max + pad)
    ax3.legend(loc="upper right", fontsize=9)
    
    def update(i):
        lineH.set_data(x_grid, H_arr[i])
        lineInv.set_data(x_grid, H_arr[i]) # H_arr est l'invariant ici
        lineTh.set_data(x_grid, th_arr[i])
        
        water_poly[0].remove()
        water_poly[0] = ax1.fill_between(x_grid, z, H_arr[i], color="deepskyblue", alpha=0.5)
        
        time_text.set_text(f"t = {times[i]:.3f} s")
        return lineH, lineInv, lineTh, water_poly[0], time_text

    anim = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
    gif_path = os.path.join(output_dir, "Cas 1 wb pertu.gif")
    
    try:
        print(f"Sauvegarde en cours vers {gif_path} ...")
        anim.save(gif_path, writer=PillowWriter(fps=15))
        print("Succès ! GIF sauvegardé.")
    except Exception as e:
        print(f"Erreur : {e}")
    
    print("Sauvegarde de l'image finale (PNG)...")
    last_frame_index = len(times) - 1
    update(last_frame_index)
    png_path = os.path.join(output_dir, "Cas 1 wb pertu.png")
    fig.savefig(png_path, dpi=200)
    print(f"Succès ! Image sauvegardée : {png_path}")

if __name__ == "__main__":
    times, H_hist, th_hist, z, H0_val, theta0_val = simulate()
    create_gif(times, H_hist, th_hist, z, H0_val, theta0_val, x)