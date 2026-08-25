import numpy as np
import os

def spherique_vers_cartesien(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z], dtype=float)

def conditions_initiales_P0_FOC(dataset_params, deck_gen_params, facility_spec, iconfig):
    NBEAMS = facility_spec["nbeams"]
    num_perturbations = dataset_params["num_perturbations"]
    power_loss_quanta = dataset_params["power_loss_quanta"]
    r = dataset_params['target_radius']

    #### Paramètres ####
    # --- Pointage (FOC) ---
    bool_pert_FOC = False               # True: pointage bruité, False: pointage nominal
    pointing_affected_beams = range(NBEAMS)   # faisceaux concernés par le pointage
    pointing_sigma = 10                 # écart-type (mm) du bruit gaussien sur X, Y, Z

    # --- Puissance (P0) ---
    bool_pert_P0 = True                # active/désactive la perte de puissance
    random_mode = 2                      # 0: aléatoire cumulatif progressif
                                         # 1: aléatoire sans répétition, progressif
                                         # 2: liste fixe, perte uniforme

    # Paramètres spécifiques à chaque mode
    power_params = {
        0: {
            "beams_pool": range(NBEAMS),      # faisceaux tirés au sort (avec remise)
        },
        1: {
            "n_quanta": 1,                     # nb de quanta perdus par faisceau sélectionné
        },
        2: {
            "affected_beams": [0, 1, 2, 8, 9],  # faisceaux fixes affectés
            "num_quanta": [2, 2, 2, 2, 2],       # quanta perdus par faisceau (même ordre)
        },
    }

               
    #### Chargement des points focaux nominaux à partir du fichier de configuration des taches ####
    beams = {}
    with open("../facility_config_files/xavier_ico30_theta_phi_rad.txt", "r") as file:
        for i, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            theta, phi = map(float, line.split())
            beams[i] = {"theta": theta, "phi": phi}
    os.chdir("../python_scripts")
    theta = np.array([beams[i]["theta"] for i in range(NBEAMS)])
    phi = np.array([beams[i]["phi"] for i in range(NBEAMS)])
    X, Y, Z = [], [], []
    for i in range(NBEAMS):
        x, y, z = spherique_vers_cartesien(r, theta[i], phi[i])
        X.append(x); Y.append(y); Z.append(z)

    # ---- Pointage ----
    for i in range(num_perturbations):
        if bool_pert_FOC:
            for j in pointing_affected_beams:
                dx = np.random.normal(0, pointing_sigma)
                dy = np.random.normal(0, pointing_sigma)
                dz = np.random.normal(0, pointing_sigma)
                deck_gen_params['pointings'][iconfig, j, i] = (X[j] + dx, Y[j] + dy, Z[j] + dz)
        else:
            for j in pointing_affected_beams:
                deck_gen_params['pointings'][iconfig, j, i] = (X[j], Y[j], Z[j])

    # ---- Puissance ----
    if bool_pert_P0:

        cfg = power_params[random_mode]

        if random_mode == 0:
            for k in range(num_perturbations - 1):
                beam_idx = np.random.choice(cfg["beams_pool"])
                deck_gen_params["p0"][:, beam_idx, :, k+1:] -= power_loss_quanta

        elif random_mode == 1: # pour avoir l'ensemble des NBEAMS affectés il faut num_perturbation = NBEAMS + 1
            shuffled_beams = np.random.permutation(NBEAMS)
            selected_beams = shuffled_beams[:num_perturbations - 1]
            for k, beam_idx in enumerate(selected_beams):
                deck_gen_params["p0"][:, beam_idx, :, k+1:] -= (cfg["n_quanta"] * power_loss_quanta)

        elif random_mode == 2:
            for beam, n_quanta in zip(cfg["affected_beams"], cfg["num_quanta"]):
                deck_gen_params["p0"][:, beam, :, :] -= n_quanta * power_loss_quanta

        # sécurité : pas de puissance négative
        deck_gen_params["p0"] = np.maximum(deck_gen_params["p0"], 0.0)

    return













