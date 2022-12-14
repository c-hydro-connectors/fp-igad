#!/usr/bin/python3

"""
igad opChain - Calculate quantiles and tresholds

__date__ = '20221202'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'igad'

General command line:
python3 igad_calculate_tresholds.py -settings_file igad_calculate_tresholds.json -domain IGAD_D2

Version(s):
20221202 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import xarray as xr
import os, json, logging, time
from argparse import ArgumentParser
import datetime as dt
import pandas as pd
import numpy as np
import lmoments3 as lm
import geopandas as gpd
import rasterio as rio
from scipy.special import gamma

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_name = 'IGAD - CALCULATE TRESHOLDS'
alg_version = '1.0.0'
alg_release = '2022-12-02'
# Algorithm parameter(s)
time_format = '%Y%m%d%H%M'

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, domain = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(logger_file=os.path.join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))

    # If domain is not provided as argument, reads it from the settings file
    if domain is None:
        domain = data_settings["data"]["domain"]
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Manage paths
    ancillary_path = data_settings["data"]["dynamic"]["ancillary"]["folder"].format(domain=domain)
    out_maps_folder = data_settings["data"]["dynamic"]["outcome"]["quantile_map_folder"].format(domain=domain)
    out_shape_folder = data_settings["data"]["dynamic"]["outcome"]["shapefile_folder"].format(domain=domain)

    os.makedirs(ancillary_path, exist_ok=True)
    annual_max_filename = data_settings["data"]["dynamic"]["ancillary"]["annual_max_filename"].format(domain=domain)
    quantile_map_filename = data_settings["data"]["dynamic"]["outcome"]["quantile_map_filename"].format(domain=domain)
    shape_out_filename = data_settings["data"]["dynamic"]["outcome"]["shapefile_filename"].format(domain=domain)

    annual_max_file = os.path.join(ancillary_path, annual_max_filename)
    quantile_map = os.path.join(out_maps_folder, quantile_map_filename)
    shape_out = os.path.join(out_shape_folder, shape_out_filename)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info(' ============================================================================ ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> START ... ')
    logging.info(' --> Domain: ' + domain)
    logging.info(' ')

    # Time algorithm information
    start_time = time.time()

    if data_settings["algorithm"]["flags"]["extract_continuum_results"]:
        logging.info(" --> Extract Continuum results and calculate annual max series...")
        start_time = dt.strptime(data_settings["data"]["dynamic"]["fp"]["start_date"], "%Y-%m-%d %H:%M")
        end_time = dt.datetime(data_settings["data"]["dynamic"]["fp"]["end_date"], "%Y-%m-%d %H:%M")
        file_now_template = os.path.join(data_settings["data"]["dynamic"]["fp"]["hmc_output_folder"], data_settings["data"]["dynamic"]["fp"]["hmc_output_filename"])
        time_range = pd.date_range(start_time, end_time, freq='H')

        year_now = start_time.year
        first_step = True
        for time_now in time_range:
            file_now = file_now_template.format(domain=str(domain), datetime_folder=time_now.strftime("%Y/%m/%d"), datetime_file=time_now.strftime("%Y%m%d%H%M"))
            logging.info(" ---> Compute time step: " + time_now.strftime("%Y%m%d%H%M"))
            os.system("gunzip -k " + file_now + ".gz >/dev/null 2>&1")
            try:
                step = xr.open_dataset(file_now)
            except:
                continue
            if first_step is True:
                ds = xr.DataArray(dims=["time", "lat", "lon"], coords={"time":np.unique(time_range.year), "lon":np.unique(step["Longitude"].values), "lat":np.unique(step["Latitude"].values)})
                ds.loc[year_now,:,:] = step["Discharge"].values
                first_step = False
            if year_now < time_now.year:
                year_now = year_now + 1
                ds.loc[year_now,:,:] = step["Discharge"].values
            else:
                ds.loc[year_now,:,:] = np.maximum(ds.loc[year_now].values,step["Discharge"].values)
            os.system("rm " + file_now)

        ds.to_netcdf(annual_max_file)
        logging.info(" --> Extract Continuum results and calculate annual max series...DONE!")
    else:
        if os.path.isfile(annual_max_file):
            logging.info(" --> Annual max file is provided, skipping extraction of Continuum results!")
        else:
            logging.error(" ERROR! Provide an annual max file or activate extraction from Continuum results!")

    if data_settings["algorithm"]["flags"]["calculate_quantile_maps"]:
        logging.info(" --> Extract quantile maps...")
        os.makedirs(out_maps_folder, exist_ok=True)

        df_annual_max = xr.open_dataset(annual_max_file)
        choice_dom = np.flipud(xr.open_rasterio(data_settings["data"]["static"]["choice"].format(domain=str(domain))).squeeze())
        annual_max = df_annual_max[data_settings["data"]["dynamic"]["ancillary"]["annual_max_varname"]].values
        logging.info(' ---> Calculate lmoments')
        [l1, l2, tau] = np.apply_along_axis(func1d=lm.lmom_ratios, axis=0, arr=annual_max, nmom=3)
        l1[choice_dom <= 0] = np.nan
        l2[choice_dom <= 0] = np.nan
        tau[choice_dom <= 0] = np.nan
        logging.info(' ---> Fitting gev params')
        params = calculate_gev_par(l1, l2, tau)
        for T in data_settings["data"]["dynamic"]["outcome"]["quantile_map_return_period"]:
            logging.info(' ---> Calculate quantile map for T=' + str(T))
            Q_T = calculate_discharge(T, params)
            if data_settings["algorithm"]["flags"]["force_positive_q"]:
                Q_T = np.where(Q_T < 0, np.nanmin(np.ma.masked_less_equal(annual_max, 0.0, copy=False), 0), Q_T)

            out_filename = quantile_map.format(domain=str(domain), T=str(T)).replace(".","p").replace("ptif", ".tif")
            out_ds = xr.DataArray(Q_T.astype('float32'), dims=["y", "x"],
                                  coords={"y": df_annual_max.lat.values, "x": df_annual_max.lon.values})
            out_ds.rio.write_crs("epsg:4326", inplace=True).rio.write_nodata(0, inplace=True).rio.to_raster(out_filename,
                                                                                                            driver="GTiff",
                                                                                                            height=len(df_annual_max.lat.values),
                                                                                                            witdh=len(df_annual_max.lon.values))
        logging.info(" --> Extract quantile maps...DONE")

    if data_settings["algorithm"]["flags"]["assign_shapefile_treshold"]:
        logging.info(" --> Assign alert tresholds to shapefile...")
        os.makedirs(out_shape_folder, exist_ok=True)
        shape = data_settings["data"]["static"]["shapefile"].format(domain=str(domain)).format(domain=domain)
        pointData = gpd.read_file(shape)

        for label, RP in enumerate(data_settings["data"]["dynamic"]["outcome"]["shapefile_th_return_period"], start=1):
            raster = quantile_map.format(domain=str(domain), T=str(T)).replace(".","p").replace("ptif", ".tif")
            if os.path.isfile(raster):
                ndviRaster = rio.open(raster)
            else:
                logging.error("ERROR! Map " + raster + " not found!")
                raise FileNotFoundError

            for ind, point in zip(pointData.index, pointData['geometry']):
                x = point.xy[0][0]
                y = point.xy[1][0]
                row, col = ndviRaster.index(x, y)
                pointData.loc[ind, "Q_THR" + str(label)] = ndviRaster.read(1)[row, col]

        pointData.to_file(shape_out)
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
    parser_handle.add_argument('-domain', action="store", dest="domain")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.domain:
        domain = parser_values.domain
    else:
        domain = None

    return alg_settings, domain

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

# -------------------------------------------------------------------------------------
# Function for calculating GEV parameters with L-moments
def calculate_gev_par(l1,l2,tau):
    c = (2/(3+tau))-(np.log(2)/np.log(3))
    params = {}
    params["theta3"] = 7.8590 * c + 2.9554 * (c ** 2)
    params["theta2"] = (l2 * params["theta3"])/((1-2 ** (-params["theta3"]))*gamma(1+params["theta3"]))
    params["theta1"] = l1 - (params["theta2"]/params["theta3"]) * (1-gamma(1+params["theta3"]))
    return params

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function for calculate design discharge with a GEV distribution
def calculate_discharge(T,params):
    P = 1 - (1/T)
    Q = np.where(params["theta3"] == 0, params["theta1"] - params["theta2"] * np.log(-np.log(P)), params["theta1"] + (params["theta2"]/params["theta3"]) * (1-(-np.log(P))**params["theta3"]))
    return Q

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------