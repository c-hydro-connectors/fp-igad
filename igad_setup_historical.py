"""
tool - IGAD - Check probabilistic
__date__ = '20221121'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python bulletin_hydro_fp.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20221121 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import os
import shutil

import datetime
import logging
import json
from argparse import ArgumentParser
import time
import pandas as pd
from copy import deepcopy

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Hydrological warning with FloodProofs '
    alg_version = '1.0.0'
    alg_release = '2022-11-22'
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, rst_time, frc_time, event = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(logger_file=os.path.join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info(' ============================================================================ ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> START ... ')
    logging.info(' ')

    if event is not None:
        data_settings["data"]["event_name"] = event

    # Time algorithm information
    start_time = time.time()
    forecast_time = datetime.datetime.strptime(frc_time, '%Y-%m-%d %H:%M')
    start_sim_time = forecast_time - datetime.timedelta(hours=data_settings["data"]["dynamic"]["time"]["obs_warmup_hours"])
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    launch_string = ''
    for domain in data_settings["data"]["domains"]:
        logging.info(" --> Setting-up domain : " + domain)

        run_settings = {}
        run_settings["hmc_flags"] = {}
        run_settings["domain"] = domain

        for flag in data_settings["data"]["static"]["hmc_flags"]:
            if data_settings["data"]["static"]["hmc_flags"][flag]:
                run_settings["hmc_flags"][flag] = 1
            else:
                run_settings["hmc_flags"][flag] = 0

        paths_filled = {}
        for key in data_settings["data"]["dynamic"]["paths"].keys():
            paths_filled[key] = data_settings["data"]["dynamic"]["paths"][key].format_map(SafeDict(domain=domain, event_name=data_settings["data"]["event_name"], frc_model=data_settings["data"]["frc_model"]))
        for key in data_settings["data"]["static"]["paths"].keys():
            paths_filled[key] = data_settings["data"]["static"]["paths"][key].format_map(SafeDict(domain=domain, event_name=data_settings["data"]["event_name"], frc_model=data_settings["data"]["frc_model"]))
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Check for the availability of a restart file
        logging.info(" --> Set up of model restart...")
        if rst_time is not None:
            restart_time = datetime.datetime.strptime(rst_time, '%Y-%m-%d %H:%M')
            logging.info(" --> Restart time manually set to : " + restart_time.strftime("%Y-%m-%d %H:%M"))
        else:
            logging.info(" --> Check for restart file in the previous 20 days...")

            for time_now in pd.date_range(start_sim_time, start_sim_time - datetime.timedelta(days=20), freq="-1H"):
                paths_rst_now = fill_path_dates(data_settings['algorithm']['template'], paths_filled, time_now)
                rst_grid = os.path.join(paths_rst_now["restart_grid"], "hmc.state-grid." + time_now.strftime("%Y%m%d%H%M") + ".nc.gz")
                rst_point = os.path.join(paths_rst_now["restart_point"], "hmc.state-point." + time_now.strftime("%Y%m%d%H%M") + ".txt")
                if os.path.isfile(rst_grid):
                    restart_time = deepcopy(time_now)
                    logging.info( ' ---> Restart file... FOUND!')
                    logging.info(' ---> Restart time set to: ' + time_now.strftime("%Y-%m-%d %H:%M"))

                    os.makedirs(os.path.join(paths_filled["working_dir"],"archive","fp_state_restart",domain,"gridded",""), exist_ok=True)
                    try:
                        shutil.copy2(rst_grid, os.path.join(paths_filled["working_dir"],"archive","fp_state_restart",domain,"gridded",""))
                    except shutil.SameFileError:
                        logging.info( " --> WARNING! Restart file are in the correct folder!")

                    if os.path.isfile(rst_point):
                        os.makedirs(
                            os.path.join(paths_filled["working_dir"], "archive", "fp_state_restart", domain, "point",""), exist_ok=True)
                        try:
                            shutil.copy2(rst_point, os.path.join(paths_filled["working_dir"], "archive", "fp_state_restart", domain, "point", ""))
                        except shutil.SameFileError:
                            logging.info(" --> WARNING! Restart file are in the correct folder!")
                    break

            if restart_time is None:
                if data_settings['algorithm']['flags']['raise_error_if_no_restart']:
                    logging.error( " --> ERROR! No valid restart file has been found!")
                    raise FileNotFoundError
                else:
                    logging.info(" --> No valid restart file has been found! Starting with a cold start!")
        logging.info(" --> Set up of model restart... DONE")
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Set paths to the actual forecast time
        logging.info(" --> Set up working environment...")
        paths_filled_now = fill_path_dates(data_settings['algorithm']['template'], paths_filled, forecast_time)
        os.makedirs(paths_filled_now["working_dir"], exist_ok=True)
        os.makedirs(os.path.join(paths_filled_now["working_dir"], "runner", ""), exist_ok=True)
        run_settings["paths"] = paths_filled_now
        run_settings["event_name"] = data_settings["data"]["event_name"]
        logging.info(" --> Set up working environment... DONE")
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # If needed, setup a states run
        if restart_time is not None:
            logging.info(" --> Set up of states run...")
            run_settings["obs_time"] = data_settings["data"]["dynamic"]["time"]["obs_warmup_hours"] + ((start_sim_time - restart_time).total_seconds()  - 86400 )/ 3600
            if not run_settings["obs_time"] == 0:
                run_settings["frc_time"] = 0
                run_settings["run_type"] = data_settings["data"]["obs_model"]
                config_algorithm, config_dataset = prepare_hmc_configuration_files(run_settings, data_settings["algorithm"]["configuration_templates"])
                launch_string = launch_string + '\npython3 $script_hmc_file -settings_algorithm ' + config_algorithm + ' -settings_datasets ' + config_dataset + ' -time "$time_now"'
                logging.info(" --> Set up of states run...DONE!")
            else:
                raise NotImplementedError(" --> Devo implementare il caso in cui il frc inizi proprio a cavallo del restart") # Devo copiare direttamente il restart
        else:
            logging.info(" --> A cold start will be performed! No state run needed!")
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Setup forecast run
        logging.info(" --> Set up of forecast run...")

        run_settings["obs_time"] = data_settings["data"]["dynamic"]["time"]["obs_warmup_hours"]
        run_settings["frc_time"] = 120
        run_settings["run_type"] = data_settings["data"]["frc_model"]
        config_algorithm, config_dataset = prepare_hmc_configuration_files(run_settings, data_settings["algorithm"]["configuration_templates"])

        launch_string = launch_string + '\npython3 $script_hmc_file -settings_algorithm ' + config_algorithm + ' -settings_datasets ' + config_dataset + ' -time "$time_now"'
        logging.info(" --> Set up of states run...DONE!")
        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
    # Setup impact based forecast run
    logging.info(" --> Set up of impact based forecast run...")
    launch_string = launch_string + "\n\nsource deactivate\nsource activate ${virtualenv_hyde_name}\n"
    run_settings["domains"] = data_settings["data"]["domains"]
    run_settings["paths"]["static_folder"] = data_settings["data"]["static"]["paths"]["static_folder"]

    for run_settings["config_ibf_template"] in data_settings["algorithm"]["configuration_templates"]["ibf_configuration"]:
        with open(os.path.join(data_settings["algorithm"]["configuration_templates"]["ibf_settings_folder"], run_settings["config_ibf_template"]), "r") as f:
            settings_ibf = json.loads(f.read().replace("{working_dir}", run_settings["paths"]["working_dir"]))
            config_ibf = compile_config_ibf(settings_ibf, run_settings)
            launch_string = launch_string + '\npython3 $script_bulletin_file -settings_file ' + config_ibf + ' -time "$time_now"'
    logging.info(" --> Set up of impact based forecast run...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Setup forecast run
    logging.info(" --> Create run launcher...")
    write_launcher(paths_filled["working_dir"], data_settings["data"]["event_name"], forecast_time, launch_string)
    logging.info(" --> Create run launcher... DONE!")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    time_elapsed = round(time.time() - start_time, 1)

    logging.info(' ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME ELAPSED: ' + str(time_elapsed) + ' seconds')
    logging.info(' ==> ... END')
    logging.info(' ==> Bye, Bye')
    logging.info(' ============================================================================ ')
    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def prepare_hmc_configuration_files(run_settings, hmc_config_templates):
    run_settings["fp_settings_folder"] = hmc_config_templates["fp_settings_folder"].format_map(
        SafeDict(event_name=run_settings["event_name"], run_type=run_settings["run_type"]))
    run_settings["config_algorithm_template"] = hmc_config_templates["fp_configuration_algorithm"].format_map(
        SafeDict(event_name=run_settings["event_name"], run_type=run_settings["run_type"]))
    run_settings["config_datasets_template"] = hmc_config_templates["fp_configuration_datasets"].format_map(
        SafeDict(event_name=run_settings["event_name"], run_type=run_settings["run_type"]))
    with open(os.path.join(run_settings["fp_settings_folder"], run_settings["config_algorithm_template"]), "r") as f:
        settings_dset = json.loads(f.read().replace("{working_dir}", run_settings["paths"]["working_dir"]).replace("{forcing_obs_dir}", run_settings["paths"]["forcing_obs_dir"]))
        config_algorithm = compile_config_algoritm(settings_dset, run_settings)
    with open(os.path.join(run_settings["fp_settings_folder"], run_settings["config_datasets_template"]), "r") as f:
        settings_dset = json.loads(f.read().replace("{working_dir}", run_settings["paths"]["working_dir"]).replace("{forcing_obs_dir}", run_settings["paths"]["forcing_obs_dir"]))
        config_dataset = compile_config_dataset(settings_dset, run_settings)
    return config_algorithm, config_dataset
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to modify a ibf configuration dataset
def compile_config_ibf(settings_dset, run_settings):
    settings_dset["data"]["dynamic"]["fp"]["static_data_folder"]= os.path.join(run_settings["paths"]["static_folder"], "gridded", "")
    settings_dset["data"]["dynamic"]["fp"]["domains"] = run_settings["domains"]
    out_name = os.path.join(run_settings["paths"]["working_dir"], "runner", run_settings["config_ibf_template"].format(domain=run_settings["domain"]))
    with open(out_name, "w") as outfile:
        json.dump(settings_dset, outfile, indent=2)
    return out_name
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to modify a hmc configuration algorithm
def compile_config_algoritm(settings_dset, run_settings):
    settings_dset["Time_Info"]["time_observed_period"] = run_settings["obs_time"]
    settings_dset["Time_Info"]["time_forecast_period"] = run_settings["frc_time"]
    settings_dset["Run_Info"]["run_type"]["run_domain"] = run_settings["domain"]
    for flag in run_settings["hmc_flags"].keys():
        settings_dset["HMC_Info"]["hmc_flags"][flag] = run_settings["hmc_flags"][flag]
    out_name = os.path.join(run_settings["paths"]["working_dir"], "runner", run_settings["config_algorithm_template"].format(domain=run_settings["domain"]))
    with open(out_name, "w") as outfile:
        json.dump(settings_dset, outfile, indent=2)
    return out_name
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to modify a hmc configuration dataset
def compile_config_dataset(settings_dset, run_settings):
    settings_dset["DataGeo"]["Gridded"]["hmc_file_folder"] = os.path.join(run_settings["paths"]["static_folder"], "gridded", "")
    settings_dset["DataGeo"]["Point"]["hmc_file_folder"] = os.path.join(run_settings["paths"]["static_folder"], "point", "")
    settings_dset["DataGeo"]["Shapefile"]["hmc_file_folder"] = os.path.join(run_settings["paths"]["static_folder"], "shapefile", "")
    out_name = os.path.join(run_settings["paths"]["working_dir"], "runner", run_settings["config_datasets_template"].format(domain=run_settings["domain"]))
    with open(out_name, "w") as outfile:
        json.dump(settings_dset, outfile, indent=2)
    return out_name
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to read file json
def read_file_json(file_name):

    env_ws = {}
    for env_item, env_value in os.environ.items():
        env_ws[env_item] = env_value

    with open(file_name, "r") as file_handle:
        json_block = []
        for file_row in file_handle:

            for env_key, env_value in env_ws.items():
                env_tag = '$' + env_key
                if env_tag in file_row:
                    env_value = env_value.strip("'\\'")
                    file_row = file_row.replace(env_tag, env_value)
                    file_row = file_row.replace('//', '/')

            # Add the line to our JSON block
            json_block.append(file_row)

            # Check whether we closed our JSON block
            if file_row.startswith('}'):
                # Do something with the JSON dictionary
                json_dict = json.loads(''.join(json_block))
                # Start a new block
                json_block = []

    return json_dict
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-rst_time', action="store", dest="rst_time")
    parser_handle.add_argument('-frc_time', action="store", dest="frc_time")
    parser_handle.add_argument('-event', action="store", dest="event")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.rst_time:
        rst_time = parser_values.rst_time
    else:
        rst_time = None

    if parser_values.frc_time:
        frc_time = parser_values.frc_time
    else:
        frc_time = None

    if parser_values.event:
        event = parser_values.event
    else:
        event = None

    return alg_settings, rst_time, frc_time, event
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):

    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.INFO)

    # Open logging basic configuration
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.INFO)
    logger_handle_2.setLevel(logging.INFO)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)

    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)

# -------------------------------------------------------------------------------------

def fill_path_dates(template_empty, paths, time_now):
    dict_filled = template_empty.copy()
    paths_filled = {}
    for key in template_empty.keys():
        dict_filled[key] = time_now.strftime(template_empty[key])
    for key in paths.keys():
        paths_filled[key] = paths[key].format_map(SafeDict(**dict_filled))
    return paths_filled

def write_launcher(working_path, event_name, time_now, launch_string):
    with open(os.path.join(working_path, "launcher.sh"),"w+") as target:
        target.write('''#!/bin/bash -e
    
# -----------------------------------------------------------------------------------------
# Script information
script_name="EVENT LAUNCHER - IGAD"
script_version="1.0.0"
script_date="2021/03/25"

# Get information (-u to get gmt time)
time_now="''' + time_now.strftime("%Y-%m-%d %H:%M") + '''"
event_name=''' + event_name + '''

# -----------------------------------------------------------------------------------------
# Script settings
system_library_folder="/home/fp/library/"
script_make_historical="igad_setup_historical.py"
settings_make_historical="/home/fp/data/fp_igad/events/igad_setup_historical.json"

# -----------------------------------------------------------------------------------------
# Get file information
virtualenv_folder=${system_library_folder}"fp_virtualenv_python3/"
virtualenv_hmc_name="fp_virtualenv_python3_hmc"
virtualenv_hyde_name="fp_virtualenv_python3_hyde"

script_hmc_folder=${system_library_folder}"hmc"
script_hmc_file=${script_hmc_folder}"/apps/HMC_Model_RUN_Manager.py"
script_bulletin_folder=$system_library_folder'bulletin/'
script_bulletin_file=${script_bulletin_folder}'/hydro/bulletin_hydro_fp.py'

runner_folder="${fp_folder}/fp_tools_runner/"
# -----------------------------------------------------------------------------------------
# Activate virtualenv
export PATH="${virtualenv_folder}/bin":$PATH
source activate $virtualenv_hmc_name

# Add path to pythonpath
export PYTHONPATH="${PYTHONPATH}:$script_hmc_folder"

''' + launch_string + '''
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
echo " ==> Bye, Bye"
echo " ==================================================================================="''')

    os.system("chmod +x " + os.path.join(working_path, "launcher.sh"))
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------