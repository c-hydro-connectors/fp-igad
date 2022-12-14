"""
tool - IGAD - Smooth maps
__date__ = '20221213'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python igad_smooth_maps.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20221121 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import os
import numpy as np
import scipy as sp
import logging
import json
from argparse import ArgumentParser
import scipy.ndimage
import xarray as xr
import time
import rioxarray
import datetime

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'igad - Smooth maps '
    alg_version = '1.0.0'
    alg_release = '2022-12-13'
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, alg_time = get_args()

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

    # Time algorithm information
    start_time = time.time()

    time_run = datetime.datetime.strptime(alg_time, '%Y-%m-%d %H:%M')

    dict_empty = data_settings['algorithm']['template']
    dict_filled = dict_empty.copy()

    for key in dict_empty.keys():
        dict_filled[key] = time_run.strftime(dict_empty[key])
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    mask = xr.open_rasterio(data_settings["data"]["static"]["domain_mask"])

    for map in data_settings["data"]["dynamic"]["maps"].keys():
        logging.info(" --> Compute map " + map)

        paths_empty = data_settings["data"]["dynamic"]["maps"][map]["file"]
        paths_filled = {}

        for key in paths_empty.keys():
            paths_filled[key] = paths_empty[key].format(**dict_filled)

        try:
            logging.info(" ---> Read map")
            input_map = xr.open_rasterio(os.path.join(paths_filled["input_folder"], paths_filled["input_filename"]))
        except FileNotFoundError:
            logging.error(" ERROR! Input " + map + " map not found!")
            raise FileNotFoundError

        logging.info(" ---> Smooth map")
        output_map = input_map.copy()
        output_map.values = gaussian_filter_with_nan(input_map.copy().values, mask.reindex_like(input_map, method="nearest").values, data_settings["data"]["dynamic"]["maps"][map]["params"]["sigma"], data_settings["data"]["dynamic"]["maps"][map]["params"]["truncate"])

        logging.info(" ---> Save map")
        os.makedirs(paths_filled["output_folder"], exist_ok=True)
        output_map.rio.to_raster(os.path.join(paths_filled["output_folder"], paths_filled["output_filename"]), compress="DEFLATE")

    logging.info(" --> Compute map " + map + "...DONE")

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

def gaussian_filter_with_nan(map,mask,sigma,truncate):
    V = map.copy()
    V[np.isnan(map)] = 0
    VV = sp.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0 * map.copy() + 1
    W[np.isnan(map)] = 0
    WW = sp.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    Z = VV / WW
    Z[mask<=0] = np.nan
    return Z
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
# Function for fill a dictionary of templates
def fill_template(downloader_settings,time_now):
    empty_template = downloader_settings["templates"]
    template_filled = {}
    for key in empty_template.keys():
        template_filled[key] = time_now.strftime(empty_template[key])
    template_filled["domain"] = downloader_settings["domain"]
    return template_filled
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time', action="store", dest="alg_time")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time:
        alg_time = parser_values.alg_time
    else:
        alg_time = None

    return alg_settings, alg_time
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

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------