# Librairies
import numpy as np
import os
import subprocess
import sys
from scipy.stats import qmc
import shutil
import glob
import stat


# Modules
import utils_deck_generation as udg
import netcdf_read_write as nrw

import utils_intensity_map as uim
import healpy_pointings as hpoint



def define_system_params(data_dir):
    sys_params = {}
    sys_params["num_parallel_ifriits"] = 4 #4
    sys_params["num_openmp_parallel"] = 4 #10
    sys_params["num_ex_checkpoint"] = 4

    sys_params["run_sims"] = True
    sys_params["run_checkpoint"] = True
    sys_params["run_clean"] = False

    sys_params["root_dir"] = ".."
    sys_params["data_dir"] = data_dir
    sys_params["config_dir"] = "config_"
    sys_params["sim_dir"] = "time_"
    sys_params["pert_dir"] = "pert_"
    sys_params["figure_location"] = "plots"
    sys_params["plot_file_type"] = ".pdf"
    sys_params["bash_parallel_ifriit"] = "bash_parallel_ifriit"
    sys_params["plasma_profile_dir"] = "plasma_profiles"
    sys_params["facility_config_files_dir"] = "facility_config_files"
    sys_params["python_dir"] = "python_scripts"

    sys_params["trainingdata_filename"] = 'dataset.nc' #"training_data_and_labels.nc"
    sys_params["dataset_params_filename"] = "dataset_params.nc"
    sys_params["facility_spec_filename"] = "facility_spec.nc"
    sys_params["deck_gen_params_filename"] = "deck_gen_params.nc"
    sys_params["ifriit_binary_filename"] = "main"

    sys_params["ifriit_run_files_dir"] = "ifriit_run_files"
    sys_params["ifriit_input_name"] = "ifriit_inputs_base.txt"
    sys_params["plasma_profile_nc"] = "ifriit_1davg_input.nc"
    sys_params["ifriit_ouput_name"] = "p_in_z1z2_beam_all.nc"
    sys_params["heat_source_nc"] = "heat_source_all_beams.nc"
    sys_params["ifriit_pulse_name"] = "pulse_per_beam.txt"

    sys_params["multi_dir"] = "multi_data"
    sys_params["multi_output_ascii_filename"] = "multi_output.txt"
    sys_params["multi_input_filename"] = "multi_input.txt"
    sys_params["multi_pulse_name"] = "laser_pulse.txt"

    return sys_params


def define_dataset_params(num_examples, sys_params,
                          random_seed=12345,
                          to_amplitude=None, bm_amplitude=None, 
                          num_perturbations=None, pi_amplitude=None):
    """
    Define dataset parameters.
    
    New optional parameters for scan:
    - to_amplitude: Target offset amplitude (fraction of target radius). If None, uses default.
    - bm_amplitude: Beam mispointing amplitude (fraction of target radius). If None, uses default.
    - num_perturbations: Number of perturbation samples. If None, uses default (100).
    - pi_amplitude: Power imbalance amplitude (fraction of default power). If None, uses default.
    """
    def define_scan_parameters(dataset_params):
        dataset_params["hemisphere_symmetric"] = False

        num_variables_per_beam = 0
        # pointings
        dataset_params["pointing_bool"] = False
        dataset_params["surface_cover_radians"] = np.radians(30.0)
        if dataset_params["pointing_bool"]:
            dataset_params["theta_index"] = num_variables_per_beam
            num_variables_per_beam += 1
            dataset_params["phi_index"] = num_variables_per_beam
            num_variables_per_beam += 1
        # defocus
        dataset_params["defocus_default"] = 0.0
        dataset_params["defocus_range"] = 35.0 # mm
        dataset_params["defocus_bool"] = False
        if dataset_params["defocus_bool"]:
            dataset_params["defocus_index"] = num_variables_per_beam
            num_variables_per_beam += 1
        # quad splitting
        dataset_params["quad_split_range"] = 3.0 # multiples of angular beam seperation within port
        dataset_params["quad_split_bool"] = False
        dataset_params["quad_split_skew_bool"] = False
        if dataset_params["quad_split_bool"]:
            dataset_params["quad_split_index"] = num_variables_per_beam
            num_variables_per_beam += 1
            if dataset_params["quad_split_skew_bool"]:
                dataset_params["quad_split_skew_index"] = num_variables_per_beam
                num_variables_per_beam += 1
        # power (time-varying?)
        dataset_params["min_power"] = 0.5 # fraction of full power
        dataset_params["power_bool"] = False
        dataset_params["time_varying_pulse"] = False
        dataset_params["num_powers_per_cone"] = 1
        if dataset_params["power_bool"]:
            dataset_params["power_index"] = num_variables_per_beam
            if dataset_params["time_varying_pulse"]:
                dataset_params["num_powers_per_cone"] = dataset_params["num_profiles_per_config"]
                num_variables_per_beam += dataset_params["num_profiles_per_config"]
            else:
                num_variables_per_beam += 1
        # beamspot
        dataset_params["select_beamspot_bool"] = True

        ##### IMPORTANTS #####
        dataset_params["beamspot_order_default"] = 2.8 #3.6 xavier ico80 60+20
        dataset_params["beamspot_radius_default"] = dataset_params['target_radius']*0.78
        dataset_params["compensation_perte_faisceau"] = True
        ######################

        dataset_params["scan_beamspot_bool"] = False
        dataset_params["beamspot_order_max"] = dataset_params["beamspot_order_default"] * 2.0
        dataset_params["beamspot_radius_min"] = dataset_params["beamspot_radius_default"] * 0.25
        dataset_params["beamspot_radius_max"] = dataset_params["beamspot_radius_default"] * 1.2
        if (dataset_params["select_beamspot_bool"] or dataset_params["scan_beamspot_bool"]):
            dataset_params["beamspot_bool"] = True
        else:
            dataset_params["beamspot_bool"] = False
        if (dataset_params["select_beamspot_bool"] and dataset_params["scan_beamspot_bool"]):
            sys.exit("Either scan or select a specific beamspot. Can't do both.")
        if dataset_params["scan_beamspot_bool"]:
            dataset_params["beamspot_order_index"] = num_variables_per_beam
            num_variables_per_beam += 1
            dataset_params["beamspot_radius_index"] = num_variables_per_beam
            num_variables_per_beam += 1
        # bandwidth
        dataset_params["select_bandwidth_bool"] = False
        dataset_params["bandwidth_num_spectral_lines_default"] = 20
        dataset_params["bandwidth_percentage_width_default"] = 1.
        dataset_params["scan_bandwidth_bool"] = False
        dataset_params["bandwidth_num_spectral_lines_max"] = dataset_params["bandwidth_num_spectral_lines_default"] * 2
        dataset_params["bandwidth_percentage_width_max"] = dataset_params["bandwidth_percentage_width_default"] * 10.0
        if (dataset_params["scan_bandwidth_bool"] or dataset_params["select_bandwidth_bool"]):
            dataset_params["bandwidth_bool"] = True
        else:
            dataset_params["bandwidth_bool"] = False
        if (dataset_params["bandwidth_bool"] and not dataset_params["run_with_cbet"]):
            sys.exit("Trying to run bandwidth without CBET?!")
        if (dataset_params["scan_bandwidth_bool"] and dataset_params["select_bandwidth_bool"]):
            sys.exit("Either scan or select a specific bandwidth. Can't do both.")
        if dataset_params["scan_bandwidth_bool"]:
            dataset_params["bandwidth_lines_index"] = num_variables_per_beam
            num_variables_per_beam += 1
            dataset_params["bandwidth_percentage_index"] = num_variables_per_beam
            num_variables_per_beam += 1

        dataset_params["num_variables_per_beam"] = num_variables_per_beam
        return dataset_params
    
    dataset_params = {}
    dataset_params["facility"] = "custom_facility" #"custom_facility" #"nif" #"lmj" # "omega"
    dataset_params["num_examples"] = num_examples
    dataset_params["random_seed"] = random_seed
    dataset_params["sampling_method"] = "linear" #"random", "lhs", "linear"
    dataset_params["run_with_cbet"] = False
    dataset_params["run_plasma_profile"] = False

    dataset_params['target_radius'] = 1940.0 #2307.0
    dataset_params['default_power'] = 1.0 # default power per beam TW

    dataset_params["plasma_profile_source"] = "default" #"multi" # "default"
    dataset_params["num_profiles_per_config"] = 1 #4
    dataset_params["plasma_profile_times"] = np.linspace(0.5,14.,int(dataset_params["num_profiles_per_config"]))
    dataset_params['illumination_evaluation_radii'] = np.zeros((dataset_params["num_profiles_per_config"])) \
                                                     + dataset_params['target_radius']

    dataset_params["imap_nside"] = 256
    dataset_params["LMAX"] = 30
    dataset_params["num_coeff"] = int(((dataset_params["LMAX"] + 2) * (dataset_params["LMAX"] + 1))/2.0)

    # Number of perturbations - use argument if provided, otherwise default
    dataset_params["num_perturbations"] = num_perturbations if num_perturbations is not None else 100
    
    # Target offset - use argument if provided, otherwise default
    if to_amplitude is not None:
        dataset_params["target_offset_bool"] = to_amplitude > 0
        dataset_params["target_offset_amplitude_mean"] = to_amplitude
    else:
        dataset_params["target_offset_bool"] = False
        dataset_params["target_offset_amplitude_mean"] = 0.01  # fraction of target radius
    
    # Beam mispointing - use argument if provided, otherwise default
    if bm_amplitude is not None:
        dataset_params["beam_mispointing_bool"] = bm_amplitude > 0
        dataset_params["beam_mispointing_amplitude_mean"] = bm_amplitude
    else:
        dataset_params["beam_mispointing_bool"] = False
        dataset_params["beam_mispointing_amplitude_mean"] = 0.05  # fraction of target radius
    
    # Power imbalance - use argument if provided, otherwise default
    dataset_params["power_loss_quanta"] = dataset_params['default_power']/32
    if pi_amplitude is not None:
        dataset_params["power_imbalance_bool"] = pi_amplitude > 0
        dataset_params["power_imbalance_amplitude_mean"] = pi_amplitude
        
    else:
        dataset_params["power_imbalance_bool"] = False
        dataset_params["power_imbalance_amplitude_mean"] = 0.05  # fraction of default power
    
    # If no perturbations are enabled (TO=0, BM=0, PI=0), only need 1 case
    if (not dataset_params["target_offset_bool"] and 
        not dataset_params["beam_mispointing_bool"] and 
        not dataset_params["power_imbalance_bool"]):
        dataset_params["num_perturbations"] = 1
        print("No perturbations enabled (TO=0, BM=0, PI=0) -> setting num_perturbations=1")
    
    # Xavier_ico80 specific parameters for dual beam radii
    # These will be used if facility is "custom_facility" with xavier_ico80$
    
    dataset_params["xavier_beam_radius_fraction_group2"] = 0.78  # 78% of target
    dataset_params["xavier_beam_radius_fraction_group1"] = dataset_params["xavier_beam_radius_fraction_group2"] / np.sqrt(1.187)  # R20/sqrt(1.187) = 0.78/sqrt(1.187) of target

    dataset_params["xavier_beam_radius_group1"] = dataset_params["xavier_beam_radius_fraction_group1"] * dataset_params['target_radius']  # 
    dataset_params["xavier_beam_radius_group2"] = dataset_params["xavier_beam_radius_fraction_group2"] * dataset_params['target_radius']  # Last 20 beams radius in microns
    
    # dataset_params["SG_MIN"]  = 4.0
    # dataset_params["SG_MAX"]  = 6.0
    # dataset_params["beamspot_order_max"]  = dataset_params["SG_MAX"]
    # The linear sampler maps [0,1] -> actual value in create_run_files_direct_drive:
    # SG  = (beamspot_order_max - 1.0) * ex_params[order_index] + 1.0
    # So beamspot_order_max = 6.0 gives SG in [1, 6] 

    #dataset_params["beamspot_radius_min"] = 0.10 * dataset_params['target_radius']
    #dataset_params["beamspot_radius_max"] = 1.20 * dataset_params['target_radius']

    dataset_params = define_scan_parameters(dataset_params)

    # facility specifications
    if dataset_params["facility"] == "nif":
        facility_spec = udg.import_nif_config(sys_params)
        # assume hemisphere symmetry
        dataset_params["num_input_params"] = int(facility_spec['num_cones']/2) * dataset_params["num_variables_per_beam"]
        dataset_params["num_beam_groups"] = int(facility_spec['num_cones']/2)
    elif (dataset_params["facility"] == "lmj") or (dataset_params["facility"] == "test"):
        facility_spec = udg.import_lmj_config(sys_params, dataset_params["quad_split_bool"])
        dataset_params["num_input_params"] = int(facility_spec['num_cones']/2) * dataset_params["num_variables_per_beam"]
        dataset_params["num_beam_groups"] = int(facility_spec['num_cones']/2)
    elif (dataset_params["facility"]=="custom_facility") or (dataset_params["facility"]=="omega"):
        facility_spec = udg.import_direct_drive_config(sys_params, dataset_params)
        dataset_params["num_beam_groups"] = 1

    dataset_params["num_input_params"] = dataset_params["num_beam_groups"] * dataset_params["num_variables_per_beam"]

    return dataset_params, facility_spec


def define_dataset(dataset_params):
    dataset = {}
    dataset["non_expand_keys"] = ["non_expand_keys","num_evaluated"]
    dataset["num_evaluated"] = 0

    dataset["input_parameters"] = np.zeros((dataset_params["num_examples"], dataset_params["num_input_params"]))
    dataset["real_modes"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"], dataset_params["num_coeff"]))
    dataset["imag_modes"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"], dataset_params["num_coeff"]))
    dataset["avg_flux"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"]))
    dataset["rms"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"]))
    dataset["power_deposited"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"]))
    dataset["power_emitted"] = np.zeros((dataset_params["num_examples"], dataset_params["num_profiles_per_config"]))
    return dataset


def populate_dataset_random_inputs(dataset_params, dataset):
    num_examples = dataset_params["num_examples"]
    num_input_params = dataset_params["num_input_params"]

    # Handle case where num_input_params is 0 (perturbation-only scan)
    if num_input_params == 0:
        print('No input parameters to sample (num_input_params=0). Skipping sampling and leaving input_parameters as zeros.')
        dataset["input_parameters"] = np.zeros((num_examples, 1))
        return dataset

    random_generator=np.random.default_rng(dataset_params["random_seed"])
    
    print(f"Sampling method: {dataset_params['sampling_method']}")

    if dataset_params["sampling_method"] == "random":
        print("Random Sampling!")
        sample = random_generator.random((dataset_params["num_examples"], dataset_params["num_input_params"]))
    elif dataset_params["sampling_method"] == "lhs":
        print("Latin Hypercube Sampling!")
        sampler = qmc.LatinHypercube(d=dataset_params["num_input_params"],
                                     strength=1, seed=random_generator, optimization="random-cd")
        sample = sampler.random(n=dataset_params["num_examples"])
    elif (dataset_params["sampling_method"] == "linear") and (num_input_params==2) \
          and (int((num_examples)**(1.0/num_input_params))**num_input_params==int(num_examples)):
        print("Linear Sampling!")
        sample = np.zeros((dataset_params["num_examples"], dataset_params["num_input_params"]))
        num_samples_per_param = int((dataset_params["num_examples"])**(1.0/dataset_params["num_input_params"]))
        val_samples_per_param = np.linspace(0.,1.,num_samples_per_param)
        iconfig = 0
        for val_param1 in val_samples_per_param:
            for val_param2 in val_samples_per_param:
                sample[iconfig, 0] = val_param1
                sample[iconfig, 1] = val_param2
                iconfig+=1
    else:
        sys.exit("dataset_params['sampling_method'] not recognised or see source code for type 'linear'")

    dataset["input_parameters"] = sample

    return dataset


def copy_python_files(sys_params):
    path_bash_file = sys_params["data_dir"]+"/"+sys_params["bash_parallel_ifriit"]
    file_exists = os.path.exists(sys_params["data_dir"]+"/"+sys_params["bash_parallel_ifriit"])
    if not file_exists:
        shutil.copy2(sys_params["root_dir"]+"/"+sys_params["bash_parallel_ifriit"],
                     path_bash_file)
    st = os.stat(path_bash_file)
    os.chmod(path_bash_file, st.st_mode | stat.S_IEXEC)

    files = glob.iglob(os.path.join(sys_params["root_dir"]+"/"+sys_params["python_dir"], "*.py"))

    file_exists = os.path.exists(sys_params["data_dir"]+"/"+sys_params["python_dir"])
    if not file_exists:
        os.makedirs( sys_params["data_dir"]+"/"+sys_params["python_dir"])

    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, sys_params["data_dir"]+"/"+sys_params["python_dir"])


def generate_training_data(dataset, dataset_params, sys_params, facility_spec):

    def run_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec):
        config_location = sys_params["data_dir"] + "/" + sys_params["config_dir"]
        for tind in range(dataset_params["num_profiles_per_config"]):
            sim_dir = "/" + sys_params["sim_dir"] + str(tind) + "/" + sys_params["pert_dir"]

            if dataset_params["run_plasma_profile"]:
                num_mpi_parallel = int(facility_spec['nbeams'] / facility_spec['beams_per_ifriit_beam'])
            else:
                num_mpi_parallel = 1

            loc_bash_parallel_ifriit = sys_params["root_dir"] + "/" + sys_params["bash_parallel_ifriit"]

            print("\n")
            print("loc_bash_parallel_ifriit: " + loc_bash_parallel_ifriit + "\n" 
                +"config_location: " + config_location + "\n" 
                + "sim_dir: " + sim_dir + "\n" 
                + "min_parallel: " + str(min_parallel) + "\n" 
                + "max_parallel: " + str(max_parallel) + "\n" 
                + "num_mpi_parallel: " + str(num_mpi_parallel) + "\n" 
                + "sys_params[\"num_openmp_parallel\"]: " + str(sys_params["num_openmp_parallel"]) + "\n" 
                + "dataset_params[\"num_perturbations\"]: " + str(dataset_params["num_perturbations"]))

            subprocess.check_call([loc_bash_parallel_ifriit, config_location, sim_dir, str(min_parallel), str(max_parallel), str(num_mpi_parallel), str(sys_params["num_openmp_parallel"]), str(dataset_params["num_perturbations"])])

        dataset = nrw.retrieve_xtrain_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)
        return dataset

    nrw.save_general_netcdf(dataset_params, sys_params["data_dir"] + "/" + sys_params["dataset_params_filename"])
    nrw.save_general_netcdf(facility_spec, sys_params["data_dir"] + "/" + sys_params["facility_spec_filename"])

    max_parallel = dataset["num_evaluated"]-1
    chkp_marker = 1.0
    filename_trainingdata = sys_params["data_dir"] + "/" + sys_params["trainingdata_filename"]

    if sys_params["run_sims"]:
        # int is a floor round
        num_parallel_runs = int((dataset_params["num_examples"] - dataset["num_evaluated"]) / sys_params["num_parallel_ifriits"])
        if num_parallel_runs > 0:
            print("aqui1")
            for ir in range(num_parallel_runs):
                min_parallel = max_parallel + 1
                max_parallel = min_parallel + sys_params["num_parallel_ifriits"] - 1
                print("aqui1")
                dataset = run_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)
                if sys_params["run_checkpoint"]:
                    if ((max_parallel + 1) >= (chkp_marker * sys_params["num_ex_checkpoint"])):
                        print("Save training data checkpoint at run: " + str(max_parallel))
                        dataset["num_evaluated"] = max_parallel + 1
                        nrw.save_general_netcdf(dataset, filename_trainingdata)
                        chkp_marker +=1

        if max_parallel != (dataset_params["num_examples"] - 1):
            print("aqui2")
            min_parallel = max_parallel + 1
            max_parallel = dataset_params["num_examples"] - 1
            dataset = run_and_delete(min_parallel, max_parallel, dataset, dataset_params, sys_params, facility_spec)

    if sys_params["run_checkpoint"]:
        print("aqui3")
        dataset["num_evaluated"] = max_parallel + 1
        nrw.save_general_netcdf(dataset, filename_trainingdata)



def main(argv):
    sys_params = define_system_params(argv[1])
    print(argv)
    
    # Parse run_type from argv[2] or argv[3]
    # Support both formats: 'full' and 'run_type=full'
    # Also support: <data_dir> <num_examples> <run_type> and <data_dir> <run_type>
    if len(argv) < 3:
        sys.exit("Usage: python training_data_generation.py <data_dir> <num_examples_OR_run_type> [run_type] [options]")
    
    # Determine if argv[2] is num_examples or run_type
    try:
        num_examples = int(argv[2])
        # argv[2] is num_examples, so run_type should be in argv[3]
        if len(argv) < 4:
            sys.exit("Usage: python training_data_generation.py <data_dir> <num_examples> <run_type> [options]")
        
        if '=' in argv[3]:
            run_type = str(argv[3]).split("=")[1]
        else:
            run_type = str(argv[3])
        
        args_start = 4
    except ValueError:
        # argv[2] is run_type (old format without num_examples)
        if '=' in argv[2]:
            run_type = str(argv[2]).split("=")[1]
        else:
            run_type = str(argv[2])
        num_examples = 10 # default value
        args_start = 3

    # Parse optional TO/BM/perturbation arguments
    num_perturbations = 1 # number of perturbation samples (if None, uses default in define_dataset_params)

    to_amplitude = None # target offset amplitude (fraction of target radius ?)
    bm_amplitude = None # beam mispointing amplitude (fraction of target radius ?)
    pi_amplitude = 0.005 # power imbalance amplitude (fraction of default power ?)

    num_parallel = None # number of parallel ifriit runs (overrides sys_params if provided)
    num_openmp = None # number of OpenMP threads (overrides sys_params if provided)
     
    for arg in argv[args_start:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key == 'to_amplitude':
                to_amplitude = float(value)
            elif key == 'bm_amplitude':
                bm_amplitude = float(value)
            elif key == 'num_perturbations':
                num_perturbations = int(value)
            elif key == 'pi_amplitude':
                pi_amplitude = float(value)
            elif key == 'num_parallel':
                num_parallel = int(value)
            elif key == 'num_openmp':
                num_openmp = int(value)
    
    # Override system params if provided
    if num_parallel is not None:
        sys_params["num_parallel_ifriits"] = num_parallel
    if num_openmp is not None:
        sys_params["num_openmp_parallel"] = num_openmp
    
    # Print info about the run
    print(f"\n{'='*60}")
    print(f"Run configuration:")
    print(f"  Data dir: {argv[1]}")
    print(f"  Num examples: {num_examples}")
    print(f"  Run type: {run_type}")

    if to_amplitude is not None:
        print(f"  TO amplitude: {to_amplitude*100:.2f}% of target radius")
    if bm_amplitude is not None:
        print(f"  BM amplitude: {bm_amplitude*100:.2f}% of target radius")
    if num_perturbations is not None:
        print(f"  Num perturbations: {num_perturbations}")
    if pi_amplitude is not None:
        print(f"  PI amplitude: {pi_amplitude*100:.2f}% of default power")

    print(f"  Num parallel ifriits: {sys_params['num_parallel_ifriits']}")
    print(f"  Num OpenMP threads: {sys_params['num_openmp_parallel']}")
    print(f"{'='*60}\n")

    if (run_type=="init") or (run_type=="full"):

        dataset_params, facility_spec = define_dataset_params(num_examples, sys_params,
            to_amplitude=to_amplitude,
            bm_amplitude=bm_amplitude,
            num_perturbations=num_perturbations,
            pi_amplitude=pi_amplitude
        )
    
        
        dataset = define_dataset(dataset_params)

        dataset = populate_dataset_random_inputs(dataset_params, dataset) # si pas de sampling alors aucune modif
        
        deck_gen_params = udg.define_deck_generation_params(dataset_params, facility_spec)

        deck_gen_params = udg.create_run_files(dataset, deck_gen_params, dataset_params, sys_params, facility_spec)

        udg.save_data_dicts_to_file(sys_params, dataset, dataset_params, deck_gen_params, facility_spec)

        copy_python_files(sys_params)

    if (run_type=="restart") or (run_type=="full"):
        dataset, dataset_params, deck_gen_params, facility_spec = udg.load_data_dicts_from_file(sys_params)
        generate_training_data(dataset, dataset_params, sys_params, facility_spec)

    return dataset, dataset_params, sys_params, facility_spec



if __name__ == "__main__":
    _, _, _, _ = main(sys.argv)