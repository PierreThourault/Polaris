# Workflow

Ce document décrit le workflow pratique pour configurer les faisceaux, générer les perturbations, lancer une campagne, et relire les résultats à partir du dictionnaire `all_tree`.

## 1) Régler `masterMind.py`

Dans `masterMind.py`, régler :
- `num_tree` : nombre d'arbres à lancer
- `main_dir` : répertoire principal où seront écrits les fichiers de sortie
- `second_dir` : chemin vers le répertoire où `masterMind.py` va relire les fichiers de sortie IFRIIT

### Attention sur les fichiers présents sur disque

A la fin de la campagne, seul **le dernier arbre** se trouvera physiquement dans le répertoire de sortie. Les arbres précédents seront écrasés.

Ce n'est pas grave :  l'entièreté des résultats  est stockée dans le dictionnaire :

- `all_tree`

Il faut donc considérer `all_tree` comme la structure de résultats de référence.

---

## 2) Structure de `all_tree`

Le premier indice est l'indice de l'arbre :

- `all_tree[indice_arbre]`

avec :
- `indice_arbre` dans `[0, num_tree[`.

Autrement dit :
- `all_tree[0]` correspond au premier arbre
- `all_tree[1]` correspond au deuxième arbre
- etc.

Dans `masterMind.py`, chaque arbre contient ensuite toutes les perturbations associées.

La structure générale est :
- `all_tree[i][j]["ifriit_inputs"]` (Paramètres faisceaux du fichier d'entrée d'IFRIIT de la perturbation j de l'arbre i)
- `all_tree[i][j]["p_in_z1z2_beam_all"]` (Champ d'intensité "brut" en sortie d'IFRIIT)
- `all_tree[i][j]["dataset"]` (Toutes les données dérivées du champ d'intensité)

avec :
- `i` = indice de l'arbre
- `j` = indice de la perturbation dans l'arbre

### Contenu des champs

#### `all_tree[i][j]["ifriit_inputs"]`
Contient les paramètres faisceau relus depuis `ifriit_inputs.txt`, faisceau par faisceau.
C'est ce champ qu'il faut utiliser pour reconstruire ce qui a réellement été injecté dans la simulation.
On y retrouve sous forme de clé :
- `P0_TW`
- `THETA_DEG`
- `PHI_DEG`
- `FOC_UM`
Exemple : `all_tree[i][j]["ifriit_inputs"][k]['THETA_DEG']` est l'angle theta du faisceau k 

#### `all_tree[i][j]["dataset"]`
Contient les quantités reconstruites à partir de la simulation, par exemple :
- `avg_flux`
- `real_modes`
- `imag_modes`
- `rms`
- `power_deposited`

#### `all_tree[i][j]["p_in_z1z2_beam_all"]`
Contient le champ d'intensité sur la sphère. Ses clés sont simplement : 'intensity', 'theta', 'phi' avec l'intensité en W/cm^2 et theta et phi en radian.
La convention utilisée est $(\theta,\varphi) \in \big[0,\pi \big] \times \big[-\pi,\pi  \big]$


### Exemples d'accès

Premier arbre, première perturbation :

```python python_scripts/WORKFLOW.md
all_tree[0][0]
```

RMS de la perturbation 3 de l'arbre 1 :

```python python_scripts/WORKFLOW.md
all_tree[1][3]["dataset"]["rms"]
```

Paramètres du faisceau 5, arbre 0, perturbation 0 :

```python python_scripts/WORKFLOW.md
beam = all_tree[0][0]["ifriit_inputs"][5]
print(beam["P0_TW"])
print(beam["THETA_DEG"])
print(beam["PHI_DEG"])
print(beam["FOC_UM"])
```

---

## 3) Régler `training_data_generation.py`

Dans `training_data_generation.py` (le main de Polaris), la variable :

- `num_perturbations`

permet de régler le **nombre de perturbations**, c'est-à-dire le **nombre de simulations dans un arbre**.

### Attention importante

Cette variable n'est active que si au moins une des trois variables ci-dessous est active :
- `to_amplitude`
- `bm_amplitude`
- `pi_amplitude`

Ces trois variables permettent d'ajouter des perturbations gausiennes à la configuration initiale.

### Rôle des trois perturbations

#### `to_amplitude`
Ajoute une perturbation de position de cible :
- TO = **target offset**

#### `bm_amplitude`
Ajoute une perturbation de pointage des faisceaux :
- BM = **beam mispointing**

#### `pi_amplitude`
Ajoute une perturbation de puissance :
- PI = **power imbalance**

### Comportement du code

Si aucune de ces perturbations n'est activée, alors le code considère qu'il n'y a pas besoin de générer plusieurs réalisations et force :

- `num_perturbations = 1`

Donc :
- demander `num_perturbations = 100` sans activer TO, BM ou PI ne produira pas 100 simulations distinctes
- il faut au moins une perturbation non nulle pour que l'arbre contienne réellement plusieurs feuilles

---

## 4) Configuration des faisceaux avec `conditions_initiales_taches.py`

La configuration personnalisée des faisceaux peut se faire à partir du script :

- `python_scripts/conditions_initiales_taches.py`

La fonction centrale est :

- `conditions_initiales_P0_FOC(dataset_params, deck_gen_params, facility_spec, iconfig)`

Cette fonction permet de modifier, pour une configuration donnée :
- les **points focaux** des faisceaux via `deck_gen_params['pointings']`
- la **puissance** des faisceaux via `deck_gen_params['p0']`

Autrement dit, si vous voulez imposer un scénario beam par beam avant l'écriture des fichiers IFRIIT, c'est dans cette fonction qu'il faut travailler.

### 4.1 Géométrie nominale utilisée

Le script lit la configuration nominale des beams dans :

- `../facility_config_files/xavier_ico30_theta_phi_rad.txt`

Chaque ligne contient :
- `theta`
- `phi`

Ces angles sont convertis en coordonnées cartésiennes à l'aide de :
- `spherique_vers_cartesien(r, theta, phi)`

avec :
- `r = dataset_params['target_radius']`

Cela définit la position focale nominale de chaque faisceau sur la cible.

### 4.2 Ce que la fonction modifie

#### A. Pointage : `deck_gen_params['pointings']`

Le code écrit de la forme :

```python python_scripts/WORKFLOW.md
deck_gen_params['pointings'][iconfig, j, i] = (x, y, z)
```

avec :
- `iconfig` = indice de configuration
- `j` = indice du faisceau
- `i` = indice de perturbation

Ce tableau contient donc le point focal 3D de chaque faisceau pour chaque perturbation.

#### B. Puissance : `deck_gen_params['p0']`

Le code modifie les puissances dans :

```python python_scripts/WORKFLOW.md
deck_gen_params["p0"][:, beam_idx, :, ...]
```

Cette structure contient les puissances par :
- configuration
- faisceau
- profil temporel
- perturbation

### 4.3 Paramètres à modifier dans `conditions_initiales_taches.py`

Dans la fonction `conditions_initiales_P0_FOC`, la zone importante est :

- `bool_pert_FOC`
- `pointing_affected_beams`
- `pointing_sigma`
- `bool_pert_P0`
- `random_mode`
- `power_params`

#### Pointage

##### `bool_pert_FOC`
- `False` : pointage nominal
- `True` : pointage bruité

##### `pointing_affected_beams`
Liste des faisceaux dont le point focal est modifié.

Exemple :

```python python_scripts/WORKFLOW.md
pointing_affected_beams = [0, 1, 2, 8, 9]
```

##### `pointing_sigma`
Écart-type du bruit gaussien ajouté à `x`, `y`, `z`.

Exemple :

```python python_scripts/WORKFLOW.md
pointing_sigma = 10
```

> Attention : les unités doivent être vérifiées avec soin. Le commentaire du script n'est pas totalement cohérent avec le reste du pipeline.

#### Puissance

##### `bool_pert_P0`
- `True` : active les pertes de puissance
- `False` : garde la puissance nominale

##### `random_mode`
Trois modes existent.

###### Mode 0 : aléatoire cumulatif avec remise

```python python_scripts/WORKFLOW.md
power_params = {
    0: {
        "beams_pool": range(NBEAMS),
    },
}
```

Effet :
- à chaque perturbation, un faisceau est tiré au hasard
- il perd un quantum de puissance
- la perte reste appliquée pour les perturbations suivantes
- un même faisceau peut être tiré plusieurs fois

###### Mode 1 : aléatoire progressif sans répétition

```python python_scripts/WORKFLOW.md
power_params = {
    1: {
        "n_quanta": 1,
    },
}
```

Effet :
- les faisceaux sont mélangés aléatoirement
- un nouveau faisceau est affecté à chaque perturbation
- pas de répétition immédiate
- la perte est cumulative

###### Mode 2 : liste fixe de faisceaux

```python python_scripts/WORKFLOW.md
power_params = {
    2: {
        "affected_beams": [0, 1, 2, 8, 9],
        "num_quanta": [2, 2, 2, 2, 2],
    },
}
```

Effet :
- les mêmes faisceaux sont toujours affectés
- chaque faisceau perd le nombre de quanta indiqué
- la perte est appliquée à toutes les perturbations

C'est le mode le plus simple pour imposer un scénario précis.

### 4.4 Exemples de scénarios

#### Cas 1 : pointage nominal, quelques beams dégradés en puissance

```python python_scripts/WORKFLOW.md
bool_pert_FOC = False
bool_pert_P0 = True
random_mode = 2

power_params = {
    2: {
        "affected_beams": [0, 1, 2, 8, 9],
        "num_quanta": [2, 2, 2, 2, 2],
    },
}
```

#### Cas 2 : bruit de pointage sur quelques beams seulement

```python python_scripts/WORKFLOW.md
bool_pert_FOC = True
pointing_affected_beams = [0, 1, 2]
pointing_sigma = 5
bool_pert_P0 = False
```

#### Cas 3 : dégradation progressive aléatoire de la puissance

```python python_scripts/WORKFLOW.md
bool_pert_FOC = False
bool_pert_P0 = True
random_mode = 1

power_params = {
    1: {
        "n_quanta": 1,
    },
}
```


### 4.5 Comment vérifier que la configuration a bien été prise en compte

Après génération, vérifier les fichiers :

- `Data/.../config_<i>/time_<t>/pert_<p>/ifriit_inputs.txt`

Dans ces fichiers, contrôler les champs :
- `P0_TW`
- `FOC_UM`
- `THETA_DEG`
- `PHI_DEG`

Ensuite, dans les résultats relus par `masterMind.py`, vérifier :

- `all_tree[i][j]["ifriit_inputs"]`

C'est ce champ qui permet de confirmer ce qui a réellement été injecté beam par beam.



