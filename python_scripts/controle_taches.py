import numpy as np
import os
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt
import pickle


def unit_vector(v):
    return v / np.linalg.norm(v)


def spherique_vers_cartesien(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z], dtype=float)


def force_moment(fi, Pi, Qi):
    u = unit_vector(Qi - Pi)
    F = fi * u
    M = np.cross(Pi, F)

    return F, M


def reconstruct_z(x, z_ref, free_idx):
    z = z_ref.copy()
    z[free_idx] = x

    return z


def Residu(x, z_ref, free_idx, Q, r):
    z = reconstruct_z(x, z_ref, free_idx)
    n = len(Q)
    f = z[:n]
    theta = z[n:2*n]
    phi = z[2*n:3*n]
    Ftot = np.zeros(3)
    Mtot = np.zeros(3)
    for i in range(n):
        Pi = spherique_vers_cartesien(r, theta[i], phi[i])
        F, M = force_moment(f[i], Pi, Q[i])
        Ftot += F
        Mtot += M

    return np.concatenate((Ftot, Mtot))


def J(x, z_ref, free_idx, Q, r, R, f0, alpha_f, alpha_P):
    z = reconstruct_z(x, z_ref, free_idx)
    n = len(Q)
    f = z[:n]
    theta = z[n:2*n]
    phi = z[2*n:3*n]
    phi_f = np.sum((f - f0)**2)
    phi_P = 0.0
    for i in range(n):
        Pi = spherique_vers_cartesien(r, theta[i], phi[i])
        Pstar = (r/R) * Q[i]
        phi_P += np.sum((Pi - Pstar)**2)

    return alpha_f * phi_f + alpha_P * phi_P



def compensation_perte_faisceau(dataset_params, deck_gen_params, facility_spec, iconfig):
    beams = {}
    with open("../facility_config_files/xavier_ico30_theta_phi_rad.txt","r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            theta, phi = map(float, line.split())
            beams[i] = {"theta": theta, "phi": phi}
    os.chdir("../python_scripts")

    n = len(beams)
    r = 1.94e-3
    R = 10.0
    Q = []
    for i in range(n):
        Q.append(spherique_vers_cartesien(R, beams[i]["theta"], beams[i]["phi"]))

    power_loss_quanta = dataset_params["power_loss_quanta"]
    nbeams = facility_spec["nbeams"]
    num_perturbations = dataset_params["num_perturbations"]

    config_initial = {}
    for k in range(num_perturbations):
        config_initial[k] = {}
        # Conditions initiales de pointage de point focal
        pointing_affected_beams = [0, 1, 2, 8, 9]
        P_theta = np.array([beams[i]["theta"] for i in range(n)])
        P_phi = np.array([beams[i]["phi"] for i in range(n)])
        # for i in pointing_affected_beams:
        #     P_theta[i] += np.pi/8 
        #     P_phi[i] += np.pi/8 

        # Conditions initiales de puissance
        power_affected_beams = [0, 1, 2, 8, 9]
        num_quanta = [1] * len(power_affected_beams)
        for i, beam in enumerate(power_affected_beams):
            deck_gen_params["p0"][:, beam, :, :] -= (num_quanta[i] * power_loss_quanta)
        f = np.squeeze(deck_gen_params["p0"][:, :, :, k])

        config_initial[k]['P0'] = f
        config_initial[k]['P_theta'] = P_theta
        config_initial[k]['P_phi'] = P_phi

        with open("results/config_initial.pkl", "wb") as file:
            pickle.dump(config_initial, file)
        
        if dataset_params["compensation_perte_faisceau"]:
    
            z_ref = np.concatenate([f, P_theta, P_phi])

            free_mask = np.ones(3 * n,dtype=bool) # on initilise le vecteur d'état avec toutes les variables libres

            for beam in pointing_affected_beams:
                free_mask[n + beam] = False
                free_mask[2*n + beam] = False
            
            for beam in power_affected_beams:
                free_mask[beam] = False


            free_idx = np.where(free_mask)[0] # permet de trouver les indices des éléments qui sont vrais

            x0 = z_ref[free_idx]

            eq_constraint = NonlinearConstraint(lambda x: Residu(x, z_ref, free_idx, Q, r), lb=np.zeros(6), ub=np.zeros(6))
            
            f0 = 1.0
            alpha_f = 1.0
            alpha_P = 1.0
            arguments = (z_ref, free_idx, Q, r, R, f0, alpha_f, alpha_P)
        
            res = minimize(J, x0, args=arguments, constraints=[eq_constraint], method="trust-constr")

            print(f"Convergence : {res.success}")
            print(f"J = {res.fun}")
            print(f"Nombre d'itérations : {res.nit}")
            print(res.message)

            z_opt = reconstruct_z(res.x, z_ref, free_idx)

            f_opt = z_opt[:n]
            theta_opt = z_opt[n:2*n]
            phi_opt = z_opt[2*n:3*n]

        else:
            f_opt = f
            theta_opt = P_theta
            phi_opt = P_phi

        deck_gen_params["p0"][0, :, 0, k] = f_opt
        for j in range(nbeams):
            deck_gen_params['pointings'][iconfig, j, k] = spherique_vers_cartesien(r, theta_opt[j], phi_opt[j])

    return