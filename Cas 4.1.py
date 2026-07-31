import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Gestion du dossier de sortie
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
output_dir = os.path.join(base_dir, "results")

g = 9.81
Nx = 300
Tfinal = 150
CFL = 0.3
xL, xR = 0.0, 20.0

A_bosse = 0.2
H_init = 0.33
H_impose=0.33
Q_inflow = 0.18
Theta_in = 1.0 
T_start_inflow = 10

# Nombre de cellules fantômes nécessaires pour WENO5
NG = 3 

def topo_z(x):
    center = (xL + xR) / 2.0
    width = 2.0
    return A_bosse * np.exp(-((x - center)**2) / (width**2))

def h_initial(x, z):
    return np.maximum(0.0, H_init - z)

def theta_initial(x):
    return Theta_in * np.ones_like(x)

# Variables globales de référence (Repos)
dx = (xR - xL) / Nx
x_grid = xL + (np.arange(Nx) + 0.5) * dx
z_ref = topo_z(x_grid)
h_ref = h_initial(x_grid, z_ref)
q_ref = np.zeros_like(x_grid)
m_ref = h_ref * theta_initial(x_grid)

# Extension des variables de référence avec fantômes (pour l'équilibre)
# On étend z_ref par continuité (Neumann plat) pour l'équilibre
z_ref_ext = np.pad(z_ref, (NG, NG), mode='edge')
h_ref_ext = np.pad(h_ref, (NG, NG), mode='edge') 
q_ref_ext = np.pad(q_ref, (NG, NG), mode='edge')
m_ref_ext = np.pad(m_ref, (NG, NG), mode='edge')


def apply_boundary_conditions_ghost(h, q, m, t):
    """
    Crée les tableaux étendus avec 3 mailles fantômes (NG=3)
    Gère les conditions physiques:
    - GAUCHE : Débit imposé progressif
    - DROITE : Sortie imposée 
    """
    # 1. Création des tableaux étendus (copie centrale)
    h_ext = np.pad(h, (NG, NG), mode='empty')
    q_ext = np.pad(q, (NG, NG), mode='empty')
    m_ext = np.pad(m, (NG, NG), mode='empty')
    
    # 2. Calcul du débit d'entrée (Rampe)
    if t < T_start_inflow:
        q_in = 0.0
    elif t < T_start_inflow + 5.0:
        ratio = (t - T_start_inflow) / 5.0
        smooth_ratio = ratio * ratio * (3 - 2 * ratio) 
        q_in = Q_inflow * smooth_ratio
    else:
        q_in = Q_inflow
        
    # BORD GAUCHE (Indices 0, 1, 2 du tableau étendu correspondent aux fantômes)
    # La première maille physique est à l'indice NG (3)
    
    # Q imposé à gauche
    q_ext[0:NG] = q_in
    
    # H Neumann à gauche (h_fantome = h_physique_0)
    h_ext[0:NG] = h[0]

    # m = h * theta
    m_ext[0:NG] = h_ext[0:NG] * Theta_in

    # BORD DROITE (Indices -3, -2, -1)
    h_ext[-NG:] = H_impose # H_imposee
    q_ext[-NG:] = q[-1]
    m_ext[-NG:] = m[-1]
    
    return h_ext, q_ext, m_ext

def flux_physique_ripa(h, q, m):
    eps = 1e-12
    h = np.maximum(h, eps); u = q / h
    p = 0.5 * g * m * h 
    return q, q * u + p, m * u

def reconstruction_weno5_with_ghost(u_ext):
    # Le tableau u_ext a une taille Nx + 2*NG
    
    # Décalages vectorisés sur le tableau étendu
    # Pour l'interface i+1/2 de la maille i (physique), on a besoin de :
    # i-2, i-1, i, i+1, i+2
    
    # u_0 correspond à la maille i. Dans u_ext, la maille physique 0 est à l'indice NG.
    # Donc on travaille sur la tranche [NG : -NG] pour u_0
    u_m2 = u_ext[NG-2 : -NG-2]
    u_m1 = u_ext[NG-1 : -NG-1]
    u_0  = u_ext[NG   : -NG]  # Cellules physiques
    u_p1 = u_ext[NG+1 : -NG+1]
    u_p2 = u_ext[NG+2 : -NG+2]
    
    # Calcul pour uL (Interface i+1/2)
    beta0 = (13/12)*(u_m2 - 2*u_m1 + u_0)**2 + 0.25*(u_m2 - 4*u_m1 + 3*u_0)**2
    beta1 = (13/12)*(u_m1 - 2*u_0 + u_p1)**2 + 0.25*(u_m1 - u_p1)**2
    beta2 = (13/12)*(u_0 - 2*u_p1 + u_p2)**2 + 0.25*(3*u_0 - 4*u_p1 + u_p2)**2

    eps_w = 1e-6
    a0 = 0.1/(eps_w+beta0)**2
    a1 = 0.6/(eps_w+beta1)**2
    a2 = 0.3/(eps_w+beta2)**2
    w_sum = a0+a1+a2

    p0 = (2*u_m2 - 7*u_m1 + 11*u_0)/6
    p1 = (-u_m1 + 5*u_0 + 2*u_p1)/6
    p2 = (2*u_0 + 5*u_p1 - u_p2)/6

    uL = (a0*p0 + a1*p1 + a2*p2)/w_sum

    # Calcul pour uR (Interface i-1/2 -> décalage pour i+1/2)
    u_p3 = u_ext[NG+3 : ] if NG+3 < len(u_ext) else u_ext[NG+3 : ]
    # Simplification : On réutilise les slices précédents mais décalés de +1
    # Pour avoir uR à l'interface i+1/2, on regarde la maille i+1 vers la gauche.
    # Stencils : {i+3, i+2, i+1}, {i+2, i+1, i}, {i+1, i, i-1}
    
    # Redéfinition propre pour uR à i+1/2 (basé sur i+1)
    # v_0 = u_{i+1}
    v_m2 = u_ext[NG-1 : -NG-1] # i-1
    v_m1 = u_ext[NG   : -NG]   # i
    v_0  = u_ext[NG+1 : -NG+1] # i+1
    v_p1 = u_ext[NG+2 : -NG+2] # i+2
    v_p2 = u_ext[NG+3 : -NG+3] # i+3 

    beta0 = (13/12)*(v_p2-2*v_p1+v_0)**2 + 0.25*(v_p2-4*v_p1+3*v_0)**2
    beta1 = (13/12)*(v_p1-2*v_0+v_m1)**2 + 0.25*(v_p1-v_m1)**2
    beta2 = (13/12)*(v_0-2*v_m1+v_m2)**2 + 0.25*(3*v_0-4*v_m1+v_m2)**2

    a0 = 0.3/(eps_w+beta0)**2; a1 = 0.6/(eps_w+beta1)**2; a2 = 0.1/(eps_w+beta2)**2
    w_sum = a0+a1+a2

    p0 = (2*v_p2 - 7*v_p1 + 11*v_0)/6
    p1 = (-v_p1 + 5*v_0 + 2*v_m1)/6
    p2 = (2*v_0 + 5*v_m1 - v_m2)/6

    uR = (a0*p0 + a1*p1 + a2*p2)/w_sum
    
    # uL et uR ont maintenant la taille (Nx)
    # Ils correspondent aux valeurs gauche/droite à l'interface i+1/2 pour i=0..Nx-1
    # Note: Pour le flux en i-1/2 (bord gauche), on aura besoin de traiter le bord flux séparément 
    # ou d'avoir calculé une interface de plus.
    # ICI : On calcule les flux aux interfaces internes + bords physiques..
    # Avec les slices ci-dessus, on obtient Nx flux (ceux à i+1/2 pour i=0..Nx-1).
    # Il manque le flux entrant à gauche (i-1/2 pour i=0).
    
    # Pour avoir Nx+1 interfaces (de gauche à droite), il faut adapter les slices pour inclure le bord gauche.
    
    return uL, uR

def reconstruction_weno5_full_interfaces(u_ext):
    # Nombre d'interfaces à calculer (Nx + 1)
    # On déduit Nx de la taille du tableau étendu (len = Nx + 2*NG)
    N_interfaces = len(u_ext) - 2 * NG + 1
    
    # 1. Reconstruction GAUCHE (uL)
    # Centrée sur la maille i=j-1 (indices décalés vers gauche)
    # Pour l'interface j, on utilise j-1 comme centre
    
    # On définit les slices par longueur explicite pour éviter les erreurs d'indices négatifs
    # Start indices pour i-2, i-1, i, i+1, i+2
    # Pour j=0 (interface gauche), centre = maille fantome NG-1.
    # i-2 correspond à NG-3.
    
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
    
    # 2. Reconstruction DROITE (uR)
    # Centrée sur la maille i=j (indices normaux)
    # Pour j=0, centre = maille physique 0 (index NG dans u_ext)
    # Stencils miroirs : on a besoin de j-2 à j+3 (indices relatifs au centre j)
    # Centre j correspond à index NG dans la boucle j=0..
    # Mais ici on veut uR pour l'interface j.
    # Pour l'interface j, le voisin de droite est la maille j (index NG+j).
    # On reconstruit uR sur la face GAUCHE de la maille j.
    # C'est équivalent à reconstruire uL sur la face DROITE de la maille j, mais avec stencil inversé.
    # Indices : i-2, i-1, i, i+1, i+2 par rapport au centre j
    
    # Start index pour le terme le plus à gauche (j-2)
    # Pour j=0, j-2 est l'index NG-2
    start_R = NG - 2
    
    v_m2 = u_ext[start_R     : start_R + N_interfaces]
    v_m1 = u_ext[start_R + 1 : start_R + 1 + N_interfaces]
    v_0  = u_ext[start_R + 2 : start_R + 2 + N_interfaces]
    v_p1 = u_ext[start_R + 3 : start_R + 3 + N_interfaces]
    v_p2 = u_ext[start_R + 4 : start_R + 4 + N_interfaces]
    
    # Formules symétriques pour uR
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
    S_star = (pR - pL + qL*(SL-uL) - qR*(SR-uR)) / (hL*(SL-uL) - hR*(SR-uR) + 1e-14)
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
    # 1. Calcul des variables totales
    h_tot = h_ref + du_h; q_tot = q_ref + du_q; m_tot = m_ref + du_m
    
    # 2. Application des Conditions aux Limites avec les cellules fqntomes
    h_ext, q_ext, m_ext = apply_boundary_conditions_ghost(h_tot, q_tot, m_tot, t)
    
    # 3. Calcul des perturbations sur le domaine étendu
    # Note : z_ref et variables de ref doivent aussi être étendues
    du_h_ext = h_ext - h_ref_ext
    du_q_ext = q_ext - q_ref_ext
    du_m_ext = m_ext - m_ref_ext
    
    # 4. Reconstruction WENO5 sur les perturbations (Nx+1 interfaces)
    dhL, dhR = reconstruction_weno5_full_interfaces(du_h_ext)
    dqL, dqR = reconstruction_weno5_full_interfaces(du_q_ext)
    dmL, dmR = reconstruction_weno5_full_interfaces(du_m_ext)
    
    # 5. États d'équilibre aux interfaces (Simple moyenne ou décalage)
    # Pour l'interface j, on prend la valeur ref de la maille j-1 (L) et j (R)
    # heL correspond à h_ref[j-1], heR à h_ref[j]
    # Attention aux bords : h_ref_ext permet de gérer ça
    heL = h_ref_ext[NG-1 : -NG]   # De l'indice -1 (fantôme gauche) à Nx-1
    heR = h_ref_ext[NG   : -NG+1] # De l'indice 0 à Nx
    qeL = q_ref_ext[NG-1 : -NG]; qeR = q_ref_ext[NG : -NG+1]
    meL = m_ref_ext[NG-1 : -NG]; meR = m_ref_ext[NG : -NG+1]
    
    # Topographie aux interfaces (pour le terme source)
    zL_int = z_ref_ext[NG-1 : -NG] 
    zR_int = z_ref_ext[NG   : -NG+1] 
    
    # Reconstruction Hydrostatique (h total)
    # hL = max(heL + dhL, 0)
    hL, hR = np.maximum(heL+dhL, 1e-12), np.maximum(heR+dhR, 1e-12)
    qL, qR = qeL+dqL, qeR+dqR
    mL, mR = meL+dmL, meR+dmR
    
    # 6. Flux Numériques (HLLC) - Taille Nx+1
    F1n, F2n, F3n = compute_hllc_flux(hL, qL, mL, hR, qR, mR)
    F1e, F2e, F3e = compute_hllc_flux(heL, qeL, meL, heR, qeR, meR)
    
    dF1, dF2, dF3 = F1n - F1e, F2n - F2e, F3n - F3e
    
    # 7. Terme Source Well-Balanced (Aux interfaces)
    S_tot = integrate_source(hL, mL, hR, mR)
    S_eq  = integrate_source(heL, meL, heR, meR)

    # Terme en (zR - zL) est défini aux interfaces
    dF2 -= 0.5 * (S_tot - S_eq) * (zR_int - zL_int)
    
    # 8. Divergence des Flux (Retour à la maille)
    # flux[i+1/2] - flux[i-1/2]
    # F[1:] est le flux en i+1/2, F[:-1] est le flux en i-1/2
    rhs_h = -(dF1[1:] - dF1[:-1])/dx
    rhs_q = -(dF2[1:] - dF2[:-1])/dx
    rhs_m = -(dF3[1:] - dF3[:-1])/dx
    
    # 9. Terme Source Volume (Centré)
    # dz/dx au centre de la maille
    # On utilise z_ref_ext pour avoir les pentes correctes aux bords si besoin
    # dz_centré_i = (z_{i+1} - z_{i-1}) / 2dx
    dz = (z_ref_ext[NG+1 : -NG+1] - z_ref_ext[NG-1 : -NG-1]) / (2*dx)
    
    h_curr = h_ref + du_h
    m_curr = m_ref + du_m
    S_vol = -g * h_curr * (m_curr / np.maximum(h_curr, 1e-12)) * dz
    S_vole = -g * h_ref * (m_ref / np.maximum(h_ref, 1e-12)) * dz
    
    rhs_q += (S_vol - S_vole)
    
    return rhs_h, rhs_q, rhs_m

def simulate_drawing():
    du_h, du_q, du_m = np.zeros_like(x_grid), np.zeros_like(x_grid), np.zeros_like(x_grid)
    times = [0.0]
    
    # Init history
    h_tot = h_ref + du_h
    m_tot = m_ref + du_m
    theta_tot = m_tot / np.maximum(h_tot, 1e-12)
    
    H_hist = [(h_tot + z_ref).copy()]
    q_hist = [(q_ref + du_q).copy()]
    th_hist = [theta_tot.copy()]
    
    t = 0.0
    next_save = 0.5
    print(f"Début simulation SCÉNARIO DESSIN (Tfinal={Tfinal}s)...")
    
    while t < Tfinal:
        # Estimation du pas de temps
        h_tot = h_ref + du_h; q_tot = q_ref + du_q; m_tot = m_ref + du_m
        c = np.sqrt(g*m_tot/np.maximum(h_tot, 1e-12))
        dt = CFL*dx/(np.max(np.abs(q_tot/h_tot)+c)+1e-12)
        if t+dt > Tfinal: dt = Tfinal-t
        
        # RK4
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
        t += dt
        
        if t >= next_save:
            times.append(t)
            h_tot = h_ref + du_h
            m_tot = m_ref + du_m
            theta_tot = m_tot / np.maximum(h_tot, 1e-12)
            
            H_hist.append((h_tot + z_ref).copy())
            q_hist.append((q_ref + du_q).copy())
            th_hist.append(theta_tot.copy())
            
            next_save += 1.0
            print(f"t={t:.1f}s")
            
    return times, H_hist, q_hist, th_hist

def create_drawing_gif(times, H_hist, q_hist, th_hist):
    print("Génération du GIF du dessin (avec Température)...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.3)
    ax1, ax2, ax3 = axs
    
    # Graph 1 : Surface et Fond
    ax1.set_title("Surface Libre et Topographie")
    ax1.fill_between(x_grid, -0.5, z_ref, color='saddlebrown', alpha=0.5, label="Fond")
    ax1.plot(x_grid, z_ref, 'k-')
    
    water_poly = [ax1.fill_between(x_grid, z_ref, H_hist[0], color="deepskyblue", alpha=0.5, label="Eau")]
    
    lineH, = ax1.plot(x_grid, H_hist[0], 'b-', lw=2, label="Surface")
    ax1.set_ylabel("Hauteur (m)")
    ax1.set_ylim(-0.1, 0.5)
    ax1.legend(loc="upper right")
    
    time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes, fontsize=12,
                         bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    # Graph 2 : Débit
    ax2.set_title(f"Débit (Injection q={Q_inflow})")
    lineQ, = ax2.plot(x_grid, q_hist[0], 'g-', lw=2)
    ax2.set_ylabel("Débit ($m^2/s$)")
    ax2.set_ylim(-0.1, Q_inflow * 2.0)
    ax2.axhline(Q_inflow, color='gray', linestyle='--', label="Consigne Entrée")
    ax2.legend(loc="upper right")

    # Graph 3 : Température
    ax3.set_title("Température (Theta)")
    lineTh, = ax3.plot(x_grid, th_hist[0], 'r-', lw=2)
    ax3.set_ylabel(r"$\theta$")
    ax3.set_xlabel("Position x (m)")
    th_min = np.min(th_hist)
    th_max = np.max(th_hist)
    pad = 0.1
    ax3.grid(True, alpha=0.3)

    def update(i):
        lineH.set_data(x_grid, H_hist[i])
        lineQ.set_data(x_grid, q_hist[i])
        lineTh.set_data(x_grid, th_hist[i])
        
        water_poly[0].remove()
        water_poly[0] = ax1.fill_between(x_grid, z_ref, H_hist[i], color="deepskyblue", alpha=0.5)

        time_text.set_text(f"t = {times[i]:.2f} s")
        
        return lineH, lineQ, lineTh, water_poly[0], time_text

    anim = FuncAnimation(fig, update, frames=len(times), interval=50)
    output_path = os.path.join(output_dir, "Cas 4.gif")
    anim.save(output_path, writer=PillowWriter(fps=15))
    print(f"GIF sauvegardé : {output_path}")

if __name__ == "__main__":
    times, H_res, q_res, th_res = simulate_drawing()
    create_drawing_gif(times, H_res, q_res, th_res)