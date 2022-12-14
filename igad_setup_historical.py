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
import numpy as np
import datetime as dt
import xarray as xr
import geopandas as gpd
from rasterio import features
from affine import Affine
import warnings
import logging
import json
from argparse import ArgumentParser
import pytz
import time
import rioxarray as rx
import pandas as pd
from copy import deepcopy
from rioxarray import merge

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
    alg_settings, alg_time = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)
    paths = {}

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(
        logger_file=os.path.join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

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