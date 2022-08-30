#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-11
# authors: jsommer

""" 
    04_calc_ndvi_ndbi.py 
    
    Calculate NDVI and NDBI remote sensing index for all given scenes.

"""

import json
import numpy as np
import rasterio as rio
from utils import read_s2_band_windowed


def scale_values(in_single_np_array, factor=10000):
    """ Scales digital number values with the given factor """
       
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    result = in_single_np_array / factor

    return result


def calc_ndvi(red_array, nir_array):
    """ Calculates NDVI index from the numpy arrays and returns the result as one dimensional numpy array.
    
    CAUTION: the following requirements have to be met:
    - input arrays have to contain Sentinel 2 L2A band digital numbers
    - all bands have to be in the same shape (i.e. resampled to 10m)

    """ 
    
    # scale digital numbers first e.g. 9999 to 0.9999 for reflectance values
    red_band = scale_values(red_array, factor=10000)
    nir_band = scale_values(nir_array, factor=10000)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    ndvi = (nir_band - red_band)/(nir_band + red_band)

    return ndvi


def calc_ndbi(swir_array, nir_array):
    """ Calculates NDBI index from the given numpy arrays and returns the result as one dimensional numpy array. 

    CAUTION: the following requirements have to be met:
    - input arrays have to contain Sentinel 2 L2A band digital numbers
    - all bands have to be in the same shape (i.e. resampled to 10m)
    
    """ 

    # scale digital numbers first e.g. 9999 to 0.9999 for reflectance values
    swir_band = scale_values(swir_array, factor=10000)
    nir_band = scale_values(nir_array, factor=10000)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    ndbi = (swir_band - nir_band)/(swir_band + nir_band)

    return ndbi


def calc_ndmi(swir_array, nir_array):
    """ Calculates NDMI index from the given numpy arrays and returns the result as one dimensional numpy array. 

    CAUTION: the following requirements have to be met:
    - input arrays have to contain Sentinel 2 L2A band digital numbers
    - all bands have to be in the same shape (i.e. resampled to 10m)
    
    """ 

    # scale digital numbers first e.g. 9999 to 0.9999 for reflectance values
    swir_band = scale_values(swir_array, factor=10000)
    nir_band = scale_values(nir_array, factor=10000)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    ndmi = (nir_band - swir_band)/(nir_band + swir_band)

    return ndmi


def calc_ndwi(green_array, nir_array):
    """ Calculates NDWI (as understood to detect water bodies) index from the given numpy arrays and returns the result as one dimensional numpy array. 

    CAUTION: the following requirements have to be met:
    - input arrays have to contain Sentinel 2 L2A band digital numbers
    - all bands have to be in the same shape (i.e. resampled to 10m)
    
    """ 

    # scale digital numbers first e.g. 9999 to 0.9999 for reflectance values
    green_band = scale_values(green_array, factor=10000)
    nir_band = scale_values(nir_array, factor=10000)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    ndwi = (green_band - nir_band)/(green_band + nir_band)

    return ndwi


def calc_ndre1(rededge_array_1, rededge_array_2):
    """ Calculates NDRE1 index from the given numpy arrays and returns the result as one dimensional numpy array. 

    (B06 - B05) / (B06 + B05)

    CAUTION: the following requirements have to be met:
    - input arrays have to contain Sentinel 2 L2A band digital numbers
    - all bands have to be in the same shape (i.e. resampled to 10m)
    
    """ 

    # scale digital numbers first e.g. 9999 to 0.9999 for reflectance values
    rededge_array_1 = scale_values(rededge_array_1, factor=10000)
    rededge_array_2 = scale_values(rededge_array_2, factor=10000)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    ndre1 = (rededge_array_2 - rededge_array_1)/(rededge_array_2 + rededge_array_1)

    return ndre1


if __name__ == '__main__':

    # duration without writing test data to disk: 0,8s

    # read test data
    geojson_file_path = "../data/osm_nominatim_Freising.geojson"
    b03, profile = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B03.tif", geojson_file_path=geojson_file_path)
    b04, profile = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B04.tif", geojson_file_path=geojson_file_path)
    b08, profile = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B08.tif", geojson_file_path=geojson_file_path)
    b11, profile = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B11.tif", geojson_file_path=geojson_file_path)

    assert b04.shape == b08.shape == b11.shape, f"shapes of bands differ: {b04.shape}, {b08.shape}, {b11.shape}"
    
    ndvi = calc_ndvi(red_array=b04, nir_array=b08)

    ndbi = calc_ndbi(swir_array=b11, nir_array=b08)

    ndwi = calc_ndwi(green_array=b03, nir_array=b08)

    assert ndvi.shape == ndbi.shape == ndwi.shape, f"shapes of ndvi, ndbi, ndwi differ: {ndvi.shape}, {ndbi.shape}, {ndwi.shape}"

    profile.update({
        "dtype": np.float32
    })

    # for testing only - write band to file system
    with rio.open(f"../../data/S2A_32UPU_20220617_0_L2A/ndvi.tif", "w", **profile) as dest:
        dest.write(ndvi)

    with rio.open(f"../../data/S2A_32UPU_20220617_0_L2A/ndbi.tif", "w", **profile) as dest:
        dest.write(ndbi)

    with rio.open(f"../../data/S2A_32UPU_20220617_0_L2A/ndwi.tif", "w", **profile) as dest:
        dest.write(ndwi)