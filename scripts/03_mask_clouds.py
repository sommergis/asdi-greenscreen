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

from utils import read_s2_band

def compute_cloudmask(scl_array, debug=False):
    ''' Converts the given Sentinel-2 L2A SCL Band to a cloud mask (0 nodata, 1 cloud, 2 no cloud) 
        and returns a numpy array
        
        see also https://earth.esa.int/documents/247904/685211/S2+L2A+ProcessingLevel+Definition+Document/2c0f6d5f-60b5-48de-bc0d-e0f45ca06304
    '''

    # pixels are classified in SCL dataset
    # 0,1,3,8,9,10: cloud or invalid pixels
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


if __name__ == '__main__':

    # read test data
    scl = read_s2_band(file_path="../../data/S2A_32UPU_20220617_0_L2A/SCL.tif", debug=True)

    clouds = compute_cloudmask(scl_array=scl)
    
    # get the rasterio profile only
    with rio.open("../../data/S2A_32UPU_20220617_0_L2A/SCL.tif", "r") as src:
        profile = src.profile

        # for testing only - write band to file system
        with rio.open(f"../../data/S2A_32UPU_20220617_0_L2A/clouds_test.tif", "w", **profile) as dest:
            dest.write(clouds)