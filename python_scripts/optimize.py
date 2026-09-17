#some code copied from https://github.com/fmfn/BayesianOptimization edited by Duncan Barlow
#some code copied from https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/master/Tutorial%20Project edited by Duncan Barlow
import training_data_generation as tdg
import netcdf_read_write as nrw
import utils_optimizers as uopt
import utils_deck_generation as idg
import numpy as np
import sys
import time
import os
import shutil
import copy
############
from unittest import result
import subprocess as sp
import xarray as xr
from scipy.optimize import minimize
import healpy as hp
import re
import time
import matplotlib.pyplot as plt


def wrapper_bayesian_optimisation(dataset, bo_params, opt_params):
    pbounds = {}
    for ii in range(opt_params["num_optimization_params"]):
        pbounds["x"+str(ii)] = opt_params["pbounds"][ii,:]

    target = uopt.fitness_function(dataset, opt_params)

    optimizer, utility = uopt.initialize_unknown_func(dataset["input_parameters"],
                                                      target, pbounds,
                                                      opt_params["num_init_examples"],
                                                      opt_params["num_optimization_params"])
    print("Starting Bayesian optimizer")

    tic = time.perf_counter()
    for it in range(opt_params["n_iter"]):
        optimizer2 = copy.deepcopy(optimizer)

        X_new = np.zeros((bo_params["ifriit_runs_per_iteration"], opt_params["num_optimization_params"]))
        old_max_eval = dataset["num_evaluated"]

        for npar in range(bo_params["ifriit_runs_per_iteration"]):
            ieval = old_max_eval + npar
            next_point = optimizer2.suggest(utility)
            for ii in range(opt_params["num_optimization_params"]):
                X_new[npar,ii] = next_point["x"+str(ii)]

            try:
                optimizer2.register(params=next_point, target=bo_params["target_set_undetermined"])
            except:
                boolean_array = np.isclose(X_new[npar,0], dataset["input_parameters"][:,0], rtol=1.0e-5)
                print("Broken input! Likely degererate, index: ", ieval)
                print("First parameter degenerate with: ", np.where(boolean_array))

                print("Trying random mutation:")
                for_mutation = X_new[npar,:].reshape((1,opt_params["num_optimization_params"]))
                print("old: ", for_mutation)
                after_mutation = uopt.mutation(for_mutation, opt_params["random_generator"],
                                           opt_params["pbounds"],
                                           num_mutations=bo_params["num_mutations"],
                                           mutation_amplitude=bo_params["mutation_amplitude"])
                print("new: ", after_mutation)
                X_new[npar,:] = after_mutation[0,:]

        old_max_eval = dataset["num_evaluated"]
        dataset = uopt.run_ifriit_input(bo_params["ifriit_runs_per_iteration"], X_new, opt_params)

        target = uopt.fitness_function(dataset, opt_params)
        for npar in range(bo_params["ifriit_runs_per_iteration"]):
            ieval = old_max_eval + npar

            for ii in range(opt_params["num_optimization_params"]):
                next_point["x"+str(ii)] = dataset["input_parameters"][ieval,ii]

            try:
                optimizer.register(params=next_point, target=target[ieval])
            except:
                print("Broken input!", next_point, target[ieval])

        if (it+1)%opt_params["printout_iteration_skip"] <= 0.0:
            uopt.printout_optimizer_iteration(tic, dataset, opt_params)
    return dataset



def wrapper_gradient_ascent(dataset, gd_params, opt_params):
    learning_rate = 10.0**gd_params["learn_exp"]
    step_size = np.array([gd_params["learn_exp"] - 1.0, gd_params["learn_exp"] + 1.0])
    stencil_size = opt_params["num_optimization_params"] * 2

    target = uopt.fitness_function(dataset, opt_params)

    X_old = np.zeros((1, opt_params["num_optimization_params"]))
    maxdex_new = np.argmax(target)
    X_old[0,:] = dataset["input_parameters"][maxdex_new,:]

    print("The index with the max fitness was: ", str(maxdex_new))
    print("It had intial rms: {:.2f} %".format(dataset["rms"][maxdex_new, 0]*100.0), " and mean intensity: {:.2e}W/sr".format(dataset["avg_flux"][maxdex_new, 0]))

    number_of_snapshots = np.shape(dataset["rms"][:,:])[1]
    if number_of_snapshots != 1:
        print("It had ablation pressure rms: {:.2f} %".format(dataset["rms"][maxdex_new, 1]*100.0), " and mean pressure: {:.2f}Mbar".format(dataset["avg_flux"][maxdex_new, 1]))

    tic = time.perf_counter()
    for ieval in range(opt_params["n_iter"]):
        maxdex_old = maxdex_new

        X_stencil = uopt.gradient_stencil(X_old, learning_rate, opt_params["pbounds"],
                                     opt_params["num_optimization_params"], stencil_size)
        dataset = uopt.run_ifriit_input(stencil_size, X_stencil, opt_params)

        target = uopt.fitness_function(dataset, opt_params)
        target_stencil = target[-stencil_size:]
        uopt.printout_optimizer_iteration(tic, dataset, opt_params)

        grad = uopt.determine_gradient(X_stencil, target_stencil, target[maxdex_old], learning_rate,
                                  opt_params["pbounds"], opt_params["num_optimization_params"])
        grad = grad / np.sum(np.abs(grad))
        X_downhill = uopt.grad_ascent(X_old, grad, step_size, opt_params["pbounds"],
                             opt_params["num_optimization_params"],
                             gd_params["num_steps_per_iter"])
        dataset = uopt.run_ifriit_input(gd_params["num_steps_per_iter"], X_downhill, opt_params)

        target = uopt.fitness_function(dataset, opt_params)
        uopt.printout_optimizer_iteration(tic, dataset, opt_params)

        maxdex_new = np.argmax(target)
        X_old[0,:] = dataset["input_parameters"][maxdex_new,:]

        if (maxdex_new == maxdex_old):
            gd_params["learn_exp"] = gd_params["learn_exp"]-0.5
            learning_rate = 10.0**(gd_params["learn_exp"])
            step_size = step_size - 0.5
            print("Reducing step size to: " + str(learning_rate))
            if learning_rate < 1.0e-4:
                print(str(ieval+1) + " Bayesian data points added, saving to .nc")
                print("Early stopping due to repeated results")
                break

        print("Iteration {} with learn rate {} value: {}".format(ieval, learning_rate, target[maxdex_new]))
        print(X_old)
    return dataset



def wrapper_genetic_algorithm(dataset, ga_params, opt_params):
    X_pop = dataset["input_parameters"]

    tic = time.perf_counter()
    for generation in range(opt_params["n_iter"]-1):
        print("Generation : ", generation+1)
        target = uopt.fitness_function(dataset, opt_params)

        # Selecting the best parents in the population for mating.
        parents = uopt.select_mating_pool(X_pop, target[-ga_params["initial_pop_size"]:], ga_params["num_parents_mating"])

        # Generating next generation using crossover.
        offspring_crossover = uopt.crossover(parents,
                                             offspring_size=(ga_params["initial_pop_size"]
                                                             - ga_params["num_parents_mating"],
                                                             opt_params["num_optimization_params"]))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = uopt.mutation(offspring_crossover, opt_params["random_generator"],
                                           opt_params["pbounds"],
                                           num_mutations=ga_params["num_mutations"],
                                           mutation_amplitude=ga_params["mutation_amplitude"])

        # Creating the new population based on the parents and offspring.
        X_pop[0:ga_params["num_parents_mating"],:] = parents
        X_pop[ga_params["num_parents_mating"]:,:] = offspring_mutation

        dataset = uopt.run_ifriit_input(ga_params["initial_pop_size"], X_pop, opt_params)

        if (generation+1)%opt_params["printout_iteration_skip"] <= 0.0:
            print(str((generation+1) * ga_params["initial_pop_size"]) + " data points added")
            uopt.printout_optimizer_iteration(tic, dataset, opt_params)
    
    return dataset


def wrapper_L_BFGS_B(x0_norm, args, bounds_norm, options, center, scale):
    history = []
    last_cost = {"value": None}

    def objective_scaled(x_norm, *args_inner):
        x_real = center + scale * x_norm
        cost = uopt.objective(x_real, *args_inner)
        last_cost["value"] = cost
        return cost

    def callback(xk):
        history.append(last_cost["value"])
        print(f"  [callback] itération {len(history)}: cost = {last_cost['value']}")

    res = minimize(objective_scaled, x0_norm, args=args, method="L-BFGS-B", bounds=bounds_norm, options=options, callback=callback)
    return res, history


def main(argv):
    """ 
                       dir         iex  init_type  bayes_opt grad_descent random_sampler random_seed  dir
    python optimize.py Data_output 100   0-2 10     0-1 10     0-1  10        0           12345      Data_input
    python optimize.py Data_output 100 2 10 1 10 1 10 0 12345 Data_input
    index:                       1  2  3 4  5  6 7 8  9   10      11
    """
    #
    data_init_type = int(argv[3])
    input_dir = argv[11]
    output_dir = argv[1]
    num_examples = int(argv[2])
    #random_seed = int(argv[10])
    #random_sampling = int(argv[9])

    sys_params = tdg.define_system_params(output_dir)

    if data_init_type == 1: # Generate new initialization dataset
        print("Generating data!")
        dataset, dataset_params, sys_params, facility_spec = tdg.main((None, sys_params["data_dir"], num_examples, "run_type=full"))

    elif data_init_type == 2: # Genetic algorithm
        print("Using a genetic algorithm!")
        ga_n_iter = int(argv[4])
        initial_pop_size = num_examples
        dataset, dataset_params, sys_params, facility_spec = tdg.main((None, sys_params["data_dir"], initial_pop_size, "run_type=full"))

        num_parents_mating = int(initial_pop_size / 10.0)
        if (num_parents_mating % 2) != 0:
            num_parents_mating -=1
        if num_parents_mating < 2:
            num_parents_mating = 2
        num_init_examples = 0 # genetic algorithm generates its own initial data

        opt_params = uopt.define_optimizer_parameters(output_dir, num_init_examples, ga_n_iter, dataset_params,
                                                     facility_spec, sys_params)
        num_mutations = int(opt_params["num_optimization_params"] / 2)

        ga_params = uopt.define_genetic_algorithm_params(initial_pop_size, num_parents_mating, num_mutations)
        dataset = wrapper_genetic_algorithm(dataset, ga_params, opt_params)

    elif data_init_type == 0:
        print("Importing pre-generated data!")
        use_L_BFGS_B = bool(int(argv[12]))
        if use_L_BFGS_B:
            source_file = os.path.join("..", "ifriit_run_files", "main")
            destination_file = os.path.join(output_dir, "main")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            shutil.copy2(source_file, destination_file)

        # copy across dataset_params and facility_spec
        shutil.copyfile(input_dir + "/" + sys_params["dataset_params_filename"],
                        output_dir + "/" + sys_params["dataset_params_filename"])
        shutil.copyfile(input_dir + "/" + sys_params["facility_spec_filename"],
                        output_dir + "/" + sys_params["facility_spec_filename"])
        shutil.copyfile(input_dir + "/" + sys_params["trainingdata_filename"],
                        output_dir + "/" + sys_params["trainingdata_filename"])
        shutil.copyfile(input_dir + "/" + sys_params["deck_gen_params_filename"],
                        output_dir + "/" + sys_params["deck_gen_params_filename"])
    else:
        print("")
        sys.exit("Dataset not properly specified")

    print("Importing data!")
    dataset_params = nrw.read_general_netcdf(sys_params["data_dir"] + "/" + sys_params["dataset_params_filename"])
    facility_spec = nrw.read_general_netcdf(sys_params["data_dir"] + "/" + sys_params["facility_spec_filename"])
    dataset = nrw.read_general_netcdf(sys_params["data_dir"] + "/" + sys_params["trainingdata_filename"])
    num_init_examples = dataset["num_evaluated"]

    use_bayesian_optimization = bool(int(argv[5]))
    if use_bayesian_optimization: # Bayesian optimization
        bo_n_iter = int(argv[6])
        opt_params = uopt.define_optimizer_parameters(output_dir, num_init_examples, bo_n_iter, dataset_params,
                                                     facility_spec, sys_params)
        ifriit_runs_per_bo_iteration = sys_params["num_parallel_ifriits"]

        target = uopt.fitness_function(dataset, opt_params)
        target_set_undetermined = np.mean(target) / 2.0 # half mean for all undetermined BO values
        num_mutations = int(opt_params["num_optimization_params"] / 2)
        bo_params = uopt.define_bayesian_optimisation_params(ifriit_runs_per_bo_iteration, target_set_undetermined, num_mutations)
        dataset = wrapper_bayesian_optimisation(dataset, bo_params, opt_params)
        num_init_examples = dataset["num_evaluated"]

    use_gradient_ascent = bool(int(argv[7]))
    if use_gradient_ascent: # Gradient ascent
        print("Using gradient ascent!")
        gd_n_iter = int(argv[8])
        line_search_evaluations = sys_params["num_parallel_ifriits"]
        opt_params = uopt.define_optimizer_parameters(output_dir, num_init_examples, gd_n_iter, dataset_params,
                                                     facility_spec, sys_params)

        gd_params = uopt.define_gradient_ascent_params(line_search_evaluations, dataset_params["num_input_params"])
        dataset = wrapper_gradient_ascent(dataset, gd_params, opt_params)
        num_init_examples = dataset["num_evaluated"]


    if use_L_BFGS_B: # L-BFGS-B
        print("Using L-BFGS-B!")

        lbfgsb_n_iter = int(argv[13])
        # Initialisation des paramètres et lecture des fichiers
        R = 1e7 # distance des miroirs en um par rapport au centre de la cible
        
        # Extraction des angles THETA et PHI à partir du fichier de configuration
        THETA = []
        PHI = []
        with open(f'../facility_config_files/{facility_spec["ifriit_facility_name"]}.txt', 'r') as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:  # ignore les lignes vides
                    continue
                valeurs = ligne.split()
                THETA.append(float(valeurs[0]))
                PHI.append(float(valeurs[1]))
        THETA = np.array(THETA)
        PHI = np.array(PHI)
        NBEAMS = len(THETA)
        
        # Lecture du fichier ifriit_inputs_originale.txt et extraction des paramètres P0, X0, Y0, Z0
        with open(input_dir +"/config_0/time_0/pert_0/ifriit_inputs.txt") as f:
            ifriit_inputs_originale = f.read()
        beam_pattern = r"&BEAM.*?/"
        beams = re.findall(beam_pattern, ifriit_inputs_originale, flags=re.DOTALL)
        if len(beams) != NBEAMS:
            raise ValueError(f"{len(beams)} beams trouvés au lieu de {NBEAMS}")

        P0_list = []
        X0_list = []
        Y0_list = []
        Z0_list = []
        for beam in beams:
            p0 = float(re.search(r"P0_TW\s*=\s*([-+0-9.eEdD]+)", beam).group(1).replace("D", "E").replace("d", "e"))

            foc = re.search(r"FOC_UM\s*=\s*([^,\n]+)\s*,\s*([^,\n]+)\s*,\s*([^,\n]+)", beam)

            x = float(foc.group(1).replace("D", "E").replace("d", "e"))
            y = float(foc.group(2).replace("D", "E").replace("d", "e"))
            z = float(foc.group(3).replace("D", "E").replace("d", "e"))

            P0_list.append(p0)
            X0_list.append(x)
            Y0_list.append(y)
            Z0_list.append(z)

        P0 = np.array(P0_list)
        X0 = np.array(X0_list)
        Y0 = np.array(Y0_list)
        Z0 = np.array(Z0_list)
        z_ref = np.concatenate([P0,np.zeros(NBEAMS), np.zeros(NBEAMS)]) # z_ref = [P0, alpha=0, beta=0]

        # Configuration de l'optimisation
        free_mask = np.ones(3 * NBEAMS, dtype=bool)
        power_affected_beams = list(range(NBEAMS))
        for beam in power_affected_beams:
            free_mask[beam] = False

        free_idx = np.where(free_mask)[0]
        x0 = z_ref[free_idx]

        bounds = []
        for i in free_idx:
            if i < NBEAMS:
                bounds.append((0.9, 1.1))  # P0
            else:
                bounds.append((-np.pi*1e-4, np.pi*1e-4))  # alpha, beta

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        scale = (hi - lo) / 2.0
        center = (hi + lo) / 2.0

        bounds_norm = [(-1.0, 1.0)] * len(bounds)

        options = {"maxiter": lbfgsb_n_iter, "gtol": 1e-9, "ftol": 1e-8}

        x0_norm = (x0 - center) / scale
        ARGS = (z_ref, free_idx, ifriit_inputs_originale, X0, Y0, Z0, THETA, PHI, R,NBEAMS,output_dir)

        t1 = time.time()
        res, history = wrapper_L_BFGS_B(x0_norm, ARGS, bounds_norm, options, center, scale)
        t2 = time.time()


        plt.figure()
        plt.plot(history)
        plt.yscale("log")
        plt.xlabel("Evaluation IFRIIT")
        plt.ylabel("Cost")
        plt.grid()
        plt.savefig("L-BFGS-B_cost_history.png")
        plt.show()

        print("Temps d'optimisation:", t2-t1, "secondes")
        print("OK:", res.success)
        print("Cost:", res.fun)
        print("Message :", res.message)
        print("Nombre d'itérations :", res.nit)
        print("Nombre d'évaluations :", res.nfev)
        print("Norme du gradient :", np.linalg.norm(res.jac))
        print(np.max(np.abs(res.jac)))

        diffs = np.diff(history)  # variation de coût entre itérations successives
        rel_diffs = np.abs(diffs) / np.maximum(np.abs(history[:-1]), 1)

        for i, (d, rd) in enumerate(zip(diffs, rel_diffs)):
            print(f"iter {i}->{i+1}: delta_f={d:.2e}, delta_f_relatif={rd:.2e}")

        z_opt = uopt.reconstruct_z(center + scale * res.x, z_ref, free_idx)
        P0 = z_opt[:NBEAMS]
        ALPHA = z_opt[NBEAMS:2*NBEAMS]
        BETA  = z_opt[2*NBEAMS:3*NBEAMS]

        X, Y, Z = uopt.rotation_sur_sphere(X0, Y0, Z0, R, THETA, PHI, ALPHA, BETA)

        os.chdir(output_dir)
        uopt.write_ifriit_input(ifriit_inputs_originale,"ifriit_inputs.txt", P0, X, Y, Z)
        sp.run(["./main"], capture_output=True, text=True)
        os.chdir("../../python_scripts")

    return


if __name__ == "__main__":
    main(sys.argv)
    print("terminated normally")
