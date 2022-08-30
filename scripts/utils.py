#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-29
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
from rasterio.warp import calculate_default_transform, Resampling, aligned_target
from rasterio.vrt import WarpedVRT
import datetime as dt
from rasterio.session import AWSSession
import boto3
import fiona
from shapely.geometry import shape
from math import floor


def read_s2_band_windowed(s2_file_path, geojson_file_path, debug=False, aws_no_sign=True, dst_crs=None):
    """ Reads one band of a Sentinel-2 scene using the bounding box of the given GeoJSON geometry (windowed read) and 
        returns a numpy array. 

        GeoJSON is supposed to be WGS84 / EPSG:4326 projection.
    
        Note that a bands coarser than 10m spatial resolution will be resampled to 10m.
    """
  
    # AWS_NO_SIGN_REQUEST
    
    env = {}
    session = None
    
    if aws_no_sign == False:
        boto3_session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
        env[session] = AWSSession(boto3_session)
    else:
        env["AWS_NO_SIGN_REQUEST"] = 1
        
    env["session"] = session
        
    with rasterio.Env(**env):
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

    if dst_crs:
        return band, profile, transformed_profile
    else:
        return band, profile


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


def mask_cloud_pixels(s2_array, cloud_array, cloud_pixel=1, nodata=np.nan):
    """ Masks out cloud pixels in given s2_array. Note that the given arrays have to fit regarding shape and dimensions """

    return np.where(cloud_array == cloud_pixel, nodata, s2_array)


def get_geojson_center(geojson_file_path):
    """ Returns the center of the first geometry """
    
    with fiona.open(geojson_file_path, "r") as src:
        g = next(iter(src))
        geom = shape(g["geometry"])
        center = geom.centroid
        
    return (center.x,center.y)
    

def get_epsg_utm_zone(coords):
    """ Returns the EPSG code (UTM Zone) for the given WGS84 coordinate tuple 
    
        CAUTION: does not handle special Norway UTM issues
    """
    
    # https://stackoverflow.com/questions/32821176/convert-from-epsg4326-to-utm-in-postgis
       
    x,y = coords
    if y > 0:
        pref = 32600
    else:
        pref = 32700
    
    zone = floor((x + 180)/6)+1
    
    return zone + pref    
    

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

    # test against AWS S3 bucket
    with open(geojson_file_path,"r") as fp:
        file_content = json.load(fp)

    geometry = file_content["features"][0]["geometry"]

    # get scenes per year

    years = [x for x in range(2017,dt.datetime.today().year+1)]

    scenes_per_year = {}

    for year in years:

        enddate = min(f"{dt.datetime.today().date()}", f"{year}-12-31")

        scenes = get_scenes(
            startdate=f"{year}-01-01", 
            enddate=f"{enddate}", 
            geojson=geometry, 
            max_cloudcover=20
        )

        scenes_per_year[year] = scenes

    # from pprint import pprint
    # pprint(scenes_per_year)

    scenes_2022 = scenes_per_year.get(2022)

    # B10 is missing for L2A
    s2_l2a_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    all_scenes = {}

    # create median stack for 2022
    for scene in scenes_2022[:1]:
        
        # dict for all bands per scene
        scene_bands = {}
        for band_id in s2_l2a_bands:
            url = scene.assets.get(band_id).get("href")

            band, profile = read_s2_band_windowed(s2_file_path=url, geojson_file_path=geojson_file_path)
            
            scene_bands[band_id] = dict(array=band, profile=profile)

        all_scenes[scene] = scene_bands

    from pprint import pprint
    pprint(all_scenes)
