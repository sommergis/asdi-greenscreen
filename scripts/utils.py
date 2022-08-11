#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-11
# authors: jsommer

""" 
    utils.py 
    
    Utility methods

"""

import json
import numpy as np
import rasterio as rio
import rasterio.features
import rasterio.mask
import rasterio.io
import os.path
from rasterio.warp import Resampling, aligned_target


def read_s2_band(file_path, debug=False):
    """ Reads one band of a Sentinel-2 scene and returns a numpy array. 
    
        Note that a band coarser than 10m spatial resolution will be resampled to 10m.
    """

    # read band
    with rio.open(file_path) as src:
        
        profile = src.profile

        transform = profile.get("transform")
        resolution_x = transform[0]

        # resample in case of coarser spatial resolution than 10m
        if resolution_x > 10:           
            scale_factor = resolution_x/10

            print(f"Resolution of {file_path} is coarser than 10m - resampling during reading band with factor {scale_factor}..")

            # resample band during read
            band = src.read(
                out_shape=(
                    src.count,
                    int(src.height * scale_factor),
                    int(src.width * scale_factor)
                ),
                resampling=Resampling.cubic
            )
            # scale image transform
            transform = src.transform * src.transform.scale(
                (src.width / band.shape[-1]),
                (src.height / band.shape[-2])
            )
            profile.update({
                'count': 1,
                'driver': 'Gtiff',
                'height': src.height * scale_factor,
                'width': src.width * scale_factor,
                'transform': transform
                }
            )

        else:
            band = src.read(
                out_shape=(
                    src.count,
                    int(src.height),
                    int(src.width)
                )
            )
            profile.update({
                'count': 1,
                'driver': 'Gtiff',
                'height': src.height,
                'width': src.width,
                'transform': src.transform
                }
            )

    if debug == True:
        # for testing only - write band to file system
        with rio.open(f"../data/{os.path.splitext(file_path)[0]}_test.tif", "w", **profile) as dest:
            dest.write(band)

    return band


if __name__ == '__main__':

    # duration without writing test data to disk: 7,7s
    # duration with writing test data to disk: 46,2s

    # read test data
    b04 = read_s2_band(file_path="../../data/S2A_32UPU_20220617_0_L2A/B04.tif")
    b08 = read_s2_band(file_path="../../data/S2A_32UPU_20220617_0_L2A/B08.tif")
    b11 = read_s2_band(file_path="../../data/S2A_32UPU_20220617_0_L2A/B11.tif")

    assert b04.shape == b08.shape == b11.shape, f"shapes of bands differ: {b04.shape}, {b08.shape}, {b11.shape}"