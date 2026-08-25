import numpy as np

def spherique_vers_cartesien(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z], dtype=float)

beams = {}

with open("../facility_config_files/xavier_ico30_theta_phi_rad.txt", "r") as f:
    for i, line in enumerate(f):
        line = line.strip() # enlève les espaces, tabulations et retours à la ligne au début et à la fin de la chaîne

        # Ignore les lignes vides
        if not line:
            continue # passse à l'itération suivante

        elements = line.split() # on split la ligne à l'espace
        theta = float(elements[0])
        phi = float(elements[1])

        beams[i] = {"theta": theta,"phi": phi}

n = len(beams)
r = 1.94e-3 #m
R = 10.0 #m
Q = []
P = []
for i in range(len(beams)):
    Q.append(spherique_vers_cartesien(R,beams[i]['theta'],beams[i]['phi']))
    P.append(spherique_vers_cartesien(r,beams[i]['theta'],beams[i]['phi']))



f = [1]*n

z = [f,P]
S = 
x = S*z

