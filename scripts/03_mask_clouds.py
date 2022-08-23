#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-11
# authors: jsommer

""" 
    03_mask_clouds.py 
    
    For all given satellite scenes reclass the L2A SCL Band for cloudy pixels and mask them out (i.e. change the pixels to NoData).

"""

import json
import numpy as np
import rasterio as rio
import rasterio.features
import rasterio.mask
import rasterio.io
import os.path

from utils import read_s2_band_windowed

def compute_cloudmask(scl_array, debug=False):
    ''' Converts the given Sentinel-2 L2A SCL Band to a cloud mask (0 nodata, 1 cloud, 2 no cloud) 
        and returns a numpy array
        
        see also https://earth.esa.int/documents/247904/685211/S2+L2A+ProcessingLevel+Definition+Document/2c0f6d5f-60b5-48de-bc0d-e0f45ca06304
    '''

    # pixels are classified in SCL dataset
    # 0,1,3,8,9,10,11: cloud or invalid pixels
    cloud_pixel_classes = { 0: True, 1: True, 2: False, 3: True, 4: False, 5: False, 6: False, 7: False, 
                            8: True, 9: True, 10: True, 11: True
                            }

    reclass_array = scl_array.copy()


    # 255 as no_data value (set from cropping SCL to geometry!)
    reclass_array[np.where( scl_array == 255 )] = 0 # no data!

    for key in list(cloud_pixel_classes.keys()):
        cloud = cloud_pixel_classes[key]
        if cloud:
            reclass_array[np.where( scl_array == key )] = 1 # cloud 
        else:
            reclass_array[np.where( scl_array == key )] = 2 # no cloud
    
    return reclass_array


def mask_cloud_pixels(s2_array, cloud_array):
    """ Masks out cloud pixels in given s2_array. Note that the given arrays have to fit regarding shape and dimensions """

    return np.where(cloud_array == 1, s2_array, np.nan)


if __name__ == '__main__':

    # read test data
    geojson_file_path = "../data/osm_nominatim_Freising.geojson"

    scl, scl_profile = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/SCL.tif", geojson_file_path=geojson_file_path)
    b04, b04_profile = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B04.tif", geojson_file_path=geojson_file_path)

    clouds = compute_cloudmask(scl_array=scl)

    s2_array_masked = mask_cloud_pixels(s2_array=b04, cloud_array=clouds)
    
    # for testing only - write band to file system
    with rio.open(f"../../data/S2A_32UPU_20220617_0_L2A/clouds.tif", "w", **scl_profile) as dest:
        dest.write(clouds)

    with rio.open(f"../../data/S2A_32UPU_20220617_0_L2A/B04_masked.tif", "w", **b04_profile) as dest:
        dest.write(s2_array_masked)