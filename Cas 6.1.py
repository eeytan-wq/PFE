import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# CONFIG
g = 9.81
Nx = 800    
xL, xR = -200.0, 200.0 

Tfinal = 35.0 
T_rupture = 5 
CFL = 0.4

# Paramètres du Barrage
x_dam = 0.0  
h_L = 6.0     
h_R = 2.0    

theta_L =1.0 
theta_R = 2.0

# Nombre de cellules fantômes
NG = 3

# Gestion du dossier de sortie
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

output_dir = os.path.join(base_dir, "results")
os.makedirs(output_dir, exist_ok=True)
print(f"Les résultats seront sauvegardés dans : {output_dir}")

def topo_z(x):
    """ Fond plat """
    return np.zeros_like(x)

def apply_boundary_conditions_ghost(h, q, m, t):
    """ 
    Crée les tableaux étendus avec 3 mailles fantômes (NG=3).
    Conditions transmissives des deux côtés.
    """
    # 1. Création des tableaux étendus
    h_ext = np.pad(h, (NG, NG), mode='edge')
    q_ext = np.pad(q, (NG, NG), mode='edge')
    m_ext = np.pad(m, (NG, NG), mode='edge') #avec le mode='edge', np.pad copie déjà la valeur du bord dans les fantômes.
    
    return h_ext, q_ext, m_ext

# État de Référence
dx = (xR - xL) / Nx
x_grid = xL + (np.arange(Nx) + 0.5) * dx

z_ref = topo_z(x_grid)
h_ref = h_R * np.ones_like(x_grid)     
q_ref = np.zeros_like(x_grid)  # Repos
m_ref = h_ref * theta_R  # m = h_R * theta_R partout

# Extension des variables de référence (Ghost Cells pour l'équilibre)
z_ref_ext = np.pad(z_ref, (NG, NG), mode='edge')
h_ref_ext = np.pad(h_ref, (NG, NG), mode='edge')
q_ref_ext = np.pad(q_ref, (NG, NG), mode='edge')
m_ref_ext = np.pad(m_ref, (NG, NG), mode='edge')

# 2. MÉTHODE NUMÉRIQUE (WENO5 + HLLC + Perturbation)

def flux_physique_ripa(h, q, m):
    eps = 1e-12
    h = np.maximum(h, eps); u = q / h
    p = 0.5 * g * m * h 
    return q, q * u + p, m * u

def reconstruction_weno5_full_interfaces(u_ext):
    """
    uL et uR aux Nx+1 interfaces.
    """
    # Nombre d'interfaces physiques à calculer : Nx + 1
    # La taille de u_ext est Nx + 2*NG
    N_interfaces = len(u_ext) - 2 * NG + 1
    
    # 1. Reconstruction GAUCHE (uL)
    # Centrée sur la maille i=j-1.
    start_L = NG - 3
    u_m2 = u_ext[start_L     : start_L + N_interfaces]
    u_m1 = u_ext[start_L + 1 : start_L + 1 + N_interfaces]
    u_0  = u_ext[start_L + 2 : start_L + 2 + N_interfaces]
    u_p1 = u_ext[start_L + 3 : start_L + 3 + N_interfaces]
    u_p2 = u_ext[start_L + 4 : start_L + 4 + N_interfaces]
    
    beta0 = (13/12)*(u_m2 - 2*u_m1 + u_0)**2 + 0.25*(u_m2 - 4*u_m1 + 3*u_0)**2
    beta1 = (13/12)*(u_m1 - 2*u_0 + u_p1)**2 + 0.25*(u_m1 - u_p1)**2
    beta2 = (13/12)*(u_0 - 2*u_p1 + u_p2)**2 + 0.25*(3*u_0 - 4*u_p1 + u_p2)**2
    
    eps_w = 1e-6
    a0 = 0.1/(eps_w+beta0)**2; a1 = 0.6/(eps_w+beta1)**2; a2 = 0.3/(eps_w+beta2)**2
    w_sum = a0+a1+a2
    p0 = (2*u_m2 - 7*u_m1 + 11*u_0)/6; p1 = (-u_m1 + 5*u_0 + 2*u_p1)/6; p2 = (2*u_0 + 5*u_p1 - u_p2)/6
    uL = (a0*p0 + a1*p1 + a2*p2)/w_sum
    
    # é. Reconstruction DROITE (uR)
    # Centrée sur la maille i=j.
    start_R = NG - 2
    v_m2 = u_ext[start_R     : start_R + N_interfaces]
    v_m1 = u_ext[start_R + 1 : start_R + 1 + N_interfaces]
    v_0  = u_ext[start_R + 2 : start_R + 2 + N_interfaces]
    v_p1 = u_ext[start_R + 3 : start_R + 3 + N_interfaces]
    v_p2 = u_ext[start_R + 4 : start_R + 4 + N_interfaces]
    
    beta0 = (13/12)*(v_p2-2*v_p1+v_0)**2 + 0.25*(v_p2-4*v_p1+3*v_0)**2
    beta1 = (13/12)*(v_p1-2*v_0+v_m1)**2 + 0.25*(v_p1-v_m1)**2
    beta2 = (13/12)*(v_0-2*v_m1+v_m2)**2 + 0.25*(3*v_0-4*v_m1+v_m2)**2
    
    a0 = 0.3/(eps_w+beta0)**2; a1 = 0.6/(eps_w+beta1)**2; a2 = 0.1/(eps_w+beta2)**2
    w_sum = a0+a1+a2
    p0 = (2*v_p2 - 7*v_p1 + 11*v_0)/6; p1 = (-v_p1 + 5*v_0 + 2*v_m1)/6; p2 = (2*v_0 + 5*v_m1 - v_m2)/6
    uR = (a0*p0 + a1*p1 + a2*p2)/w_sum
    
    return uL, uR

def compute_hllc_flux(hL, qL, mL, hR, qR, mR):
    eps = 1e-12
    uL = qL / np.maximum(hL, eps); uR = qR / np.maximum(hR, eps)
    cL = np.sqrt(np.maximum(0, g * mL)); cR = np.sqrt(np.maximum(0, g * mR))
    SL = np.minimum(uL - cL, uR - cR); SR = np.maximum(uL + cL, uR + cR)
    pL = 0.5*g*mL*hL; pR = 0.5*g*mR*hR
    
    denom = hL*(SL-uL) - hR*(SR-uR)
    S_star = (pR - pL + qL*(SL-uL) - qR*(SR-uR)) / (denom + 1e-14)
    
    phi_L = hL * (SL - uL) / (SL - S_star + eps)
    phi_R = hR * (SR - uR) / (SR - S_star + eps)
    
    hL_s, qL_s, mL_s = phi_L, phi_L*S_star, phi_L*(mL/np.maximum(hL, eps))
    hR_s, qR_s, mR_s = phi_R, phi_R*S_star, phi_R*(mR/np.maximum(hR, eps))
    
    F1L, F2L, F3L = flux_physique_ripa(hL, qL, mL)
    F1R, F2R, F3R = flux_physique_ripa(hR, qR, mR)
    F1, F2, F3 = np.zeros_like(hL), np.zeros_like(hL), np.zeros_like(hL)
    
    mask_L = (SL >= 0); mask_R = (SR <= 0)
    mask_sL = (SL < 0) & (S_star >= 0); mask_sR = (S_star < 0) & (SR > 0)
    
    F1[mask_L], F2[mask_L], F3[mask_L] = F1L[mask_L], F2L[mask_L], F3L[mask_L]
    F1[mask_R], F2[mask_R], F3[mask_R] = F1R[mask_R], F2R[mask_R], F3R[mask_R]
    F1[mask_sL] = F1L[mask_sL] + SL[mask_sL]*(hL_s[mask_sL]-hL[mask_sL])
    F2[mask_sL] = F2L[mask_sL] + SL[mask_sL]*(qL_s[mask_sL]-qL[mask_sL])
    F3[mask_sL] = F3L[mask_sL] + SL[mask_sL]*(mL_s[mask_sL]-mL[mask_sL])
    F1[mask_sR] = F1R[mask_sR] + SR[mask_sR]*(hR_s[mask_sR]-hR[mask_sR])
    F2[mask_sR] = F2R[mask_sR] + SR[mask_sR]*(qR_s[mask_sR]-qR[mask_sR])
    F3[mask_sR] = F3R[mask_sR] + SR[mask_sR]*(mR_s[mask_sR]-mR[mask_sR])
    return F1, F2, F3

def integrate_source(hL, mL, hR, mR):
    thetaL = mL/np.maximum(hL, 1e-12); thetaR = mR/np.maximum(hR, 1e-12)
    S_L = -g*hL*thetaL; S_R = -g*hR*thetaR
    h_mid = 0.5*(hL+hR); m_mid = 0.5*(mL+mR)
    S_mid = -g*h_mid*(m_mid/np.maximum(h_mid, 1e-12))
    return (S_L + 4*S_mid + S_R)/6.0

def compute_rhs(du_h, du_q, du_m, t):
    # 1. Variables totales
    h_tot = h_ref + du_h; q_tot = q_ref + du_q; m_tot = m_ref + du_m
    
    # 2. Application GHOST CELLS
    h_ext, q_ext, m_ext = apply_boundary_conditions_ghost(h_tot, q_tot, m_tot, t)
    
    # 3. Perturbations étendues
    du_h_ext = h_ext - h_ref_ext
    du_q_ext = q_ext - q_ref_ext
    du_m_ext = m_ext - m_ref_ext
    
    # 4. Reconstruction WENO5 (Nx+1 interfaces)
    dhL, dhR = reconstruction_weno5_full_interfaces(du_h_ext)
    dqL, dqR = reconstruction_weno5_full_interfaces(du_q_ext)
    dmL, dmR = reconstruction_weno5_full_interfaces(du_m_ext)
    
    # 5. Récupération de l'équilibre aux interfaces
    heL = h_ref_ext[NG-1 : -NG]
    heR = h_ref_ext[NG   : -NG+1]
    qeL = q_ref_ext[NG-1 : -NG]; qeR = q_ref_ext[NG : -NG+1]
    meL = m_ref_ext[NG-1 : -NG]; meR = m_ref_ext[NG : -NG+1]
    
    # Topographie aux interfaces (pour terme source)
    zL_int = z_ref_ext[NG-1 : -NG]
    zR_int = z_ref_ext[NG   : -NG+1]
    
    # Variables reconstruites totales
    hL, hR = np.maximum(heL+dhL, 1e-12), np.maximum(heR+dhR, 1e-12)
    qL, qR = qeL+dqL, qeR+dqR
    mL, mR = meL+dmL, meR+dmR
    
    # 6. Flux (HLLC)
    F1n, F2n, F3n = compute_hllc_flux(hL, qL, mL, hR, qR, mR)
    F1e, F2e, F3e = compute_hllc_flux(heL, qeL, meL, heR, qeR, meR)
    
    dF1, dF2, dF3 = F1n - F1e, F2n - F2e, F3n - F3e
    
    # 7. Source Well-Balanced
    S_tot = integrate_source(hL, mL, hR, mR)
    S_eq  = integrate_source(heL, meL, heR, meR)
    dF2 -= 0.5 * (S_tot - S_eq) * (zR_int - zL_int)
    
    # 8. Divergence (flux[i+1/2] - flux[i-1/2])
    rhs_h = -(dF1[1:] - dF1[:-1])/dx
    rhs_q = -(dF2[1:] - dF2[:-1])/dx
    rhs_m = -(dF3[1:] - dF3[:-1])/dx
    
    # 9. Source Volume (Centrée)
    dz = (z_ref_ext[NG+1 : -NG+1] - z_ref_ext[NG-1 : -NG-1]) / (2*dx)
    
    h_curr = h_ref + du_h
    m_curr = m_ref + du_m
    S_vol = -g * h_curr * (m_curr / np.maximum(h_curr, 1e-12)) * dz
    S_vole = -g * h_ref * (m_ref / np.maximum(h_ref, 1e-12)) * dz
    
    rhs_q += (S_vol - S_vole)
    
    return rhs_h, rhs_q, rhs_m

# 3. SIMULATION

def simulate_dam_break():
    du_h = np.zeros_like(x_grid)
    du_q = np.zeros_like(x_grid)
    du_m = np.zeros_like(x_grid)
    
    # Initialisation Dam Break (perturbation par rapport au fond plat h_R)
    # Background h_ref = h_R
    # Gauche : h = h_L -> du_h = h_L - h_R
    # Droite : h = h_R -> du_h = 0
    mask_L = x_grid < x_dam
    du_h[mask_L] = h_L - h_R
    
    # Température
    # m_ref = h_R
    # m_L = h_L -> du_m = h_L - h_R
    du_m[mask_L] = (h_L * theta_L) - (h_R * theta_R)
    
    times = [0.0]
    
    h_tot = h_ref + du_h
    m_tot = m_ref + du_m
    theta_tot = m_tot / np.maximum(h_tot, 1e-12)
    
    H_hist = [(h_tot + z_ref).copy()]
    q_hist = [(q_ref + du_q).copy()]
    th_hist = [theta_tot.copy()]
    
    t = 0.0
    next_save = 0.5
    print(f"Début simulation (Domain: [{xL}, {xR}], Gate: {x_dam})...")
    print(f"Amont: {h_L}m, Aval: {h_R}m")
    
    while t < Tfinal:
        # Estimation dt
        h_tot = h_ref + du_h; q_tot = q_ref + du_q; m_tot = m_ref + du_m
        c = np.sqrt(g*m_tot/np.maximum(h_tot, 1e-12))
        u = q_tot / np.maximum(h_tot, 1e-12)
        dt_cfl = CFL*dx/(np.max(np.abs(u)+c)+1e-12)
        dt = min(dt_cfl, 0.01) # Plafonnement sûr
        
        if t+dt > Tfinal: dt = Tfinal-t
        
        if t >= T_rupture:
            k1h, k1q, k1m = compute_rhs(du_h, du_q, du_m, t)
            du_h2 = du_h+0.5*dt*k1h; du_q2 = du_q+0.5*dt*k1q; du_m2 = du_m+0.5*dt*k1m
            k2h, k2q, k2m = compute_rhs(du_h2, du_q2, du_m2, t + 0.5*dt)
            du_h3 = du_h+0.5*dt*k2h; du_q3 = du_q+0.5*dt*k2q; du_m3 = du_m+0.5*dt*k2m
            k3h, k3q, k3m = compute_rhs(du_h3, du_q3, du_m3, t + 0.5*dt)
            du_h4 = du_h+dt*k3h; du_q4 = du_q+dt*k3q; du_m4 = du_m+dt*k3m
            k4h, k4q, k4m = compute_rhs(du_h4, du_q4, du_m4, t + dt)
            du_h += (dt/6.0)*(k1h+2*k2h+2*k3h+k4h)
            du_q += (dt/6.0)*(k1q+2*k2q+2*k3q+k4q)
            du_m += (dt/6.0)*(k1m+2*k2m+2*k3m+k4m)
        else:
            if t + dt > T_rupture: dt = T_rupture - t
            
        t += dt
        
        if t >= next_save:
            times.append(t)
            h_tot = h_ref + du_h
            m_tot = m_ref + du_m
            theta_tot = m_tot / np.maximum(h_tot, 1e-12)
            H_hist.append((h_tot + z_ref).copy())
            q_hist.append((q_ref + du_q).copy())
            th_hist.append(theta_tot.copy())
            next_save += 0.5
            print(f"t={t:.2f}s")
        

            
    return times, H_hist, q_hist, th_hist

def create_dam_break_gif(times, H_hist, q_hist, th_hist):
    print("Génération du GIF...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.3)
    ax1, ax2, ax3 = axs
    
    # Graph 1 : Hauteur
    ax1.set_title(f"Dam Break (Papier: hL={h_L}m, hR={h_R}m)")
    ax1.fill_between(x_grid, -1.0, z_ref, color='saddlebrown', alpha=0.5)
    lineH, = ax1.plot(x_grid, H_hist[0], 'b-', lw=2, label="Surface")
    # Pour update le fill_between proprement
    fill_poly = [ax1.fill_between(x_grid, z_ref, H_hist[0], color="deepskyblue", alpha=0.2)]
    dam_line = ax1.axvline(x=x_dam, color='black', linewidth=4)
    ax1.set_ylabel("Hauteur (m)")
    ax1.set_ylim(-0.5, h_L * 1.1)
    ax1.set_xlim(xL, xR)
    
    time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes, fontsize=12,
                         bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    # Graph 2 : Débit
    ax2.set_title("Débit")
    lineQ, = ax2.plot(x_grid, q_hist[0], 'g-', lw=2)
    ax2.set_ylabel("Débit ($m^2/s$)")
    ax2.set_xlim(xL, xR)
    ax2.set_ylim(-10, 50) 
    ax2.grid(True, alpha=0.3)

    # Graph 3 : Température
    ax3.set_title("Température")
    lineTh, = ax3.plot(x_grid, th_hist[0], 'r-', lw=2)
    ax3.set_ylabel(r"$\theta$")
    ax3.set_xlim(xL, xR)
    ax3.set_ylim(0.5, 2.5)
    ax3.grid(True, alpha=0.3)
    
    def update(i):
        lineH.set_data(x_grid, H_hist[i])
        lineQ.set_data(x_grid, q_hist[i])
        lineTh.set_data(x_grid, th_hist[i])
        
        if times[i] < T_rupture:
            dam_line.set_linestyle('-')
            dam_line.set_alpha(1.0)
        else:
            dam_line.set_linestyle(':')
            dam_line.set_alpha(0.2)

        fill_poly[0].remove()
        fill_poly[0] = ax1.fill_between(x_grid, z_ref, H_hist[i], color="deepskyblue", alpha=0.3)
        time_text.set_text(f"t = {times[i]:.2f} s")
        return lineH, lineQ, lineTh, fill_poly[0], time_text

    anim = FuncAnimation(fig, update, frames=len(times), interval=50)
    output_path = os.path.join(output_dir, "Cas 6.gif")
    anim.save(output_path, writer=PillowWriter(fps=15))
    print(f"GIF sauvegardé : {output_path}")

if __name__ == "__main__":
    times, H_res, q_res, th_res = simulate_dam_break()
    create_dam_break_gif(times, H_res, q_res, th_res)