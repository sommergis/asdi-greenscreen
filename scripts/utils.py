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


def read_s2_band_windowed(s2_file_path, geojson_file_path, debug=False):
    """ Reads one band of a Sentinel-2 scene using the bounding box of the given GeoJSON geometry (windowed read) and 
        returns a numpy array. 

        GeoJSON is supposed to be WGS84 / EPSG:4326 projection.
    
        Note that a bands coarser than 10m spatial resolution will be resampled to 10m.
    """

    # read GeoJSON
    with open(geojson_file_path,"r") as src:
        file_content = json.load(src)

    # read band
    with rio.open(s2_file_path) as src:
        
        profile = src.profile

        transform = profile.get("transform")
        resolution_x = transform[0]
        
        # resample to 10m first and then derive the out_window
        # because otherwise the arrays of the bands won't match later

        # resample in case of coarser spatial resolution than 10m
        if resolution_x > 10:           
            scale_factor = resolution_x/10

            print(f"Resolution of {s2_file_path} is coarser than 10m - resampling during reading band with factor {scale_factor}..")

            # transform geometry first to src's CRS
            shapes_reprojected = rasterio.warp.transform_geom(
                                    "EPSG:4326", # GeoJSON should be in 4326
                                    src.crs,
                                    [feature["geometry"] for feature in file_content.get("features")]
                                )
            # get window for GeoJSON geometry
            out_window = rio.mask.geometry_window(src, shapes_reprojected)

            # resample band during windowed read
            band = src.read(
                window=out_window,
                out_shape=(
                    src.count,
                    int(out_window.height * scale_factor),
                    int(out_window.width * scale_factor)
                ),
                resampling=Resampling.nearest
            )

            # after resampling derive the proper out_window on the 10m resampled subset
            # because otherwise the arrays of the bands won't match later

            # get affine transformation by window (& new origin)
            out_transform = src.window_transform(out_window)

            # scale out_transform
            out_transform = out_transform * out_transform.scale(
                (out_window.width / band.shape[-1]),
                (out_window.height / band.shape[-2])
            )

            # important: set width & height again to windowed / numpy array
            width = band.shape[2] 
            height = band.shape[1]
            
            # scale image transform & ensure the pixel alignment to the original pixels
            aligned_transform, width, height = rio.warp.aligned_target(
                transform=out_transform, 
                height=height,
                width=width, 
                resolution=10
            )
            
            # now clip again to 10m window - otherwise we will have a bigger raster
            # write band to memory for passing it again to geometry_window
            profile.update({
                'driver': 'Gtiff',
                'height': height,
                'width': width,
                'transform': aligned_transform
                }
            )
            with rio.io.MemoryFile() as memfile:
                with memfile.open(**profile) as dest:
                    dest.write(
                        band
                    )
                with memfile.open() as src:
                    # get window for GeoJSON geometry
                    out_window = rio.mask.geometry_window(src, shapes_reprojected)
                    
                    # read resampled 10m raster again with new window -> result will be in same shape as original 10m rasters
                    band = src.read(
                        window=out_window,
                        out_shape=(
                            src.count,
                            int(out_window.height),
                            int(out_window.width)
                        )
                    )
                    # get new transform
                    out_transform = src.window_transform(out_window)

                    # update profile again
                    profile.update({
                        'driver': 'Gtiff',
                        'height': out_window.height,
                        'width': out_window.width,
                        'transform': out_transform
                        }
                    )

        else:
            
            # transform geometry first to src's CRS
            shapes_reprojected = rasterio.warp.transform_geom(
                                    "EPSG:4326", # GeoJSON should be in 4326
                                    src.crs,
                                    [feature["geometry"] for feature in file_content.get("features")]
                                )
            # get window for GeoJSON geometry
            out_window = rio.mask.geometry_window(src, shapes_reprojected)

            # windowed read
            band = src.read(
                window=out_window,
                out_shape=(
                    src.count,
                    int(out_window.height),
                    int(out_window.width)
                )
            )
            # get affine transformation by window (& new origin)
            out_transform = src.window_transform(out_window)
           
            # scale image transform & ensure the pixel alignment to the original pixels
            aligned_transform, width, height = rio.warp.aligned_target(
                transform=out_transform, 
                height=out_window.height,
                width=out_window.width, 
                resolution=10
            )

            # set output transformation, width & height
            out_transform = aligned_transform
            width = out_window.width
            height = out_window.height

            profile.update({
                'driver': 'Gtiff',
                'height': height,
                'width': width,
                'transform': out_transform
                }
            )

    if debug == True:
        with rio.open(f"../data/{os.path.splitext(s2_file_path)[0]}_windowed", "w", **profile) as dest:
            dest.write(band)

    return band


def read_s2_band(s2_file_path, debug=False):
    """ Reads one band of a Sentinel-2 scene and returns a numpy array. 
    
        Note that a band coarser than 10m spatial resolution will be resampled to 10m.
    """

    # read band
    with rio.open(s2_file_path) as src:
        
        profile = src.profile

        transform = profile.get("transform")
        resolution_x = transform[0]

        # resample in case of coarser spatial resolution than 10m
        if resolution_x > 10:           
            scale_factor = resolution_x/10

            print(f"Resolution of {s2_file_path} is coarser than 10m - resampling during reading band with factor {scale_factor}..")

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
        with rio.open(f"../data/{os.path.splitext(s2_file_path)[0]}_test.tif", "w", **profile) as dest:
            dest.write(band)

    return band


if __name__ == '__main__':

    ##
    # Regular GeoTiff read
    ##

    # duration without writing test data to disk: 7,7s
    # duration with writing test data to disk: 46,2s

    # geojson_file_path = "../data/osm_nominatim_Freising.geojson"
    # b04 = read_s2_band(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B04.tif", debug=True)
    # b08 = read_s2_band(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B08.tif", debug=True)
    # b11 = read_s2_band(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B11.tif", debug=True)

    # assert b04.shape == b08.shape == b11.shape, f"shapes of bands differ: {b04.shape}, {b08.shape}, {b11.shape}"
    
    ##
    #  Windowed read of GeoTiff file
    ##

    # duration without writing test data to disk: 0,8s
    # duration with writing test data to disk: 1,5s

    # read test data
    geojson_file_path = "../data/osm_nominatim_Freising.geojson"
    b04 = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B04.tif", geojson_file_path=geojson_file_path, debug=False)
    b08 = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B08.tif", geojson_file_path=geojson_file_path, debug=False)
    b11 = read_s2_band_windowed(s2_file_path="../../data/S2A_32UPU_20220617_0_L2A/B11.tif", geojson_file_path=geojson_file_path, debug=False)

    assert b04.shape == b08.shape == b11.shape, f"shapes of bands differ: {b04.shape}, {b08.shape}, {b11.shape}"

