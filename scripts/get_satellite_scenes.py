#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-29
# authors: jsommer

""" 
    get_satellite_scenes.py 
    
    Get all satellite scenes for the last 5 years for the given city boundary (GeoJSON) and create median composites for each year.
    Contains also functions for creating a reprojected ESA Worldcover GeoTiff with the extent of the city boundary.

"""

from satsearch import Search
import datetime as dt

import rasterio as rio
from rasterio.features import bounds
from rasterio.plot import reshape_as_raster
from rasterio.vrt import WarpedVRT
from rasterio import shutil as rio_shutil
import json
from utils import (
    read_s2_band_windowed, 
    compute_cloudmask,
    mask_cloud_pixels
)

import numpy as np
import bottleneck as bn

#from dask.distributed import Client
#import dask.array as da

from rasterio.warp import calculate_default_transform, reproject, Resampling

def get_scenes(startdate, enddate, geojson, max_cloudcover=20):
    """ Gets Sentinel-2 scenes (L2A) for the given start & endate and the given geojson """

    print(f"Searching for scenes from {startdate} to {enddate} with max cc {max_cloudcover}..")

    # only request images with cloudcover less than 20%
    query = {
        "eo:cloud_cover": {
            "lt": max_cloudcover
            }
        }
    search = Search(
        url='https://earth-search.aws.element84.com/v0',
        intersects=geojson,
        datetime=f"{startdate}/{enddate}",
        collections=['sentinel-s2-l2a-cogs'],
        query=query
    )

    items = search.items()
    
    result = []
    for item in items:
        result.append(item)

    return result



def process_scene(scene, geojson_file_path):
    """ Helper function for parallelizing work """ 

    # dict for all bands per scene
    scene_bands = {}

    # B10 is missing for L2A
    s2_l2a_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    # read SCL only once
    url = scene.assets.get("B01").get("href")
    scl_url = url.replace("B01", "SCL")
    scl_array, scl_profile = read_s2_band_windowed(s2_file_path=scl_url, geojson_file_path=geojson_file_path)

    for band_id in s2_l2a_bands:
        url = scene.assets.get(band_id).get("href")

        band, profile = read_s2_band_windowed(s2_file_path=url, geojson_file_path=geojson_file_path)

        # cloud masking per band: mask out cloudy pixels
        cld_mask = compute_cloudmask(scl_array=scl_array)

        band = mask_cloud_pixels(s2_array=band, cloud_array=cld_mask, nodata=np.nan)

        processing_baseline = scene.properties.get("sentinel:processing_baseline", "03.01")
        
        if processing_baseline >= '04.00':
            boa_offset_applied = scene.properties.get("sentinel:boa_offset_applied", False)
            
            # apply band offset -1000
            if boa_offset_applied == False:
                band = band - 1000
                print("Applied band offset manually...")
        
        scene_bands[band_id] = dict(array=band, profile=profile)

    return scene, scene_bands



def read_scenes_composites(scenes, geojson_file_path):
    """ Reads all given scenes from AWS bucket into lists of numpy arrays """

    all_scenes = {}
    
    # parallelize with threads
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = int(len(scenes)/2)
    if max_workers < 8:
        max_workers = 8
    print("Running {0} threads for reading scene bands...".format(max_workers))

    processes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for scene in scenes:
            processes.append(executor.submit(process_scene, scene, geojson_file_path))

    for task in as_completed(processes):
        try:
            scene, scene_bands = task.result()

            all_scenes[scene] = scene_bands

        except Exception as ex:
            print(ex)

    # # get all scenes as numpy array
    # for scene in scenes:

    #     scene_bands = process_scene(scene, geojson_file_path)
    #     all_scenes[scene] = scene_bands

    return all_scenes


def stack_bands_per_scene(all_scenes):
    """ Stack the numpy bands per scene and return a dictionary with scene_id as key
        and a numpy stack with shape (rows, cols, bands)
    """ 

    tmp_scenes = {}

    print(f"Processing {len(list(all_scenes.keys()))} scenes")
    
    # step 1: stack all bands of every scene
    for scene in list(all_scenes.keys()):
      
        print(f"processing scene {scene}..")
        all_bands = all_scenes.get(scene)
        
        all_bands_arrays = []
        for band_id in all_bands.keys():
            all_bands_arrays.append(all_bands.get(band_id).get("array").squeeze().astype("uint16"))
        
        # use stack() & axis = 2 for third dimension
        # rasterio's numpy axis ordering is: bands, rows, cols
        # while other libs perfer: row, cols, bands
        all_bands_stack = np.stack(all_bands_arrays, axis=2)

        tmp_scenes[scene] = all_bands_stack
        

    # step 2: stack all scenes to a big stack
    all_scenes_arrays = []
    max_shape = ()
    for scene_id in tmp_scenes.keys():
        all_scenes_arrays.append(tmp_scenes.get(scene_id))
        scene_shape = tmp_scenes.get(scene_id).shape

        if max_shape < scene_shape:
            max_shape = scene_shape
    
    print(f"Nb. of scenes before cleaning: {len(all_scenes_arrays)}")
    
    # only keep the max shape arrays (eliminate half clipped scenes!)
    all_scenes_arrays = [s for s in all_scenes_arrays if s.shape == max_shape]

    # update profile with the correct dimensions & transform
    max_width = max_height = 0

    for s in all_scenes.keys():
        bands = all_scenes.get(s)
        for b in bands.keys():
            info = bands.get(b)
            width = info.get("profile").get("width")
            if max_width < width:
                max_width = width
            height = info.get("profile").get("height")
            if max_height < height:
                max_height = height
    
    for s in all_scenes.keys():
        bands = all_scenes.get(s)
        for b in bands.keys():
            info = bands.get(b)
            width = info.get("profile").get("width")
            if width == max_width:
                profile = info.get("profile")
    
    print(f"Nb. of scenes after cleaning: {len(all_scenes_arrays)}")
    
    if len(all_scenes_arrays) > 1:
        all_scenes_stack = np.stack(all_scenes_arrays, axis=3)

    else:
        # expand dims on single scene for median calc
        all_scenes_stack = np.expand_dims(all_scenes_arrays[0], axis=3)
       

    return all_scenes_stack, profile


def calc_median(array):
    """ Very efficient median computation (cython based) with Bottleneck package """

    return bn.nanmedian(
        np.where(array < 1, np.nan, array).astype("float32"), 
        axis=3
    ).astype("uint16")


def compute_median_stack(all_scenes_stack):
    """ Calculate the median from the given all scenes stack with shape (time, bands, rows, cols) and
        returns a median stack with shape (bands, rows, cols)
    """

    print("compute_median_stack()")
    print(all_scenes_stack.shape)
    
    rows, cols, bands, time = all_scenes_stack.shape
        
    return calc_median(all_scenes_stack)


def create_median_composites(geojson_file_path, city, data_dir, years=[2017,2018,2019,2021], max_cloudcover=40):
    """ Main function for creation of median composites for the given city geojson and years """

    with open(geojson_file_path,"r") as fp:
        file_content = json.load(fp)

    geometry = file_content["features"][0]["geometry"]

    # get scenes per quarter of year
    scenes_per_year = {}
   
    quarters = {
        1: ["-01-01", "-03-31"],
        2: ["-04-01", "-06-30"],
        3: ["-07-01", "-09-30"],
        4: ["-10-01", "-12-31"]
    }

    #
    #   main loop: for all years and all quarters do the median composite calculation
    #
    for year in years:

        output_dataset = f"{data_dir}/composites/median_{city}_{year}.tif"

        quarter_medians = []

        # calculate median quarters per year (because of RAM issues!)
        # the resulting 4 median quarters will be merged at the end to a year median composite
        for q in quarters.keys():

            startdate = f"{year}{quarters.get(q)[0]}"
            enddate =  f"{year}{quarters.get(q)[1]}"

            scenes = get_scenes(
                startdate=f"{startdate}",
                enddate=f"{enddate}", 
                geojson=geometry, 
                max_cloudcover=max_cloudcover
            )

            scenes_per_year[year] = scenes

            scenes = scenes_per_year.get(year)

            print(f"Got {len(scenes)} scenes!")

            # read the scenes from AWS bucket (windowed read with geojson) into list of numpy arrays
            all_scenes = read_scenes_composites(
                scenes=scenes, 
                geojson_file_path=geojson_file_path
            )

            # stack all bands per scene and create a stack with shape
            # (rows, cols, bands, time)
            # removes also half clipped scenes
            all_scenes_stack, profile = stack_bands_per_scene(
                all_scenes=all_scenes
            )
            print("checking for stack size..")
            
            # check for RAM
            rows, cols, bands, time = all_scenes_stack.shape
            if rows * cols * bands * time > 1500*1500*12*10:
                # make stack smaller in time dimension
                # e.g. keep every 2th scene or only allow 3 scenes
                print("reducing size of to 3 scenes")
                all_scenes_stack = all_scenes_stack[:, :, :, :3]
                
            print("computing median..")

            # create the median stack with shape (rows, cols, bands)
            quarter_median = compute_median_stack(all_scenes_stack=all_scenes_stack)

            quarter_medians.append(quarter_median)
            
            # profile update
            profile.update(
                {
                    "count": quarter_median.shape[2],
                    "driver": "GTiff",
                    "compress":"LZW",
                    "predictor":2,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "num_threads": 'all_cpus',
                }
            )

            # transform to rasterio's numpy axis order: bands, rows, cols
            quarter_median = reshape_as_raster(quarter_median)

            # write to file
            with rio.open(f"{output_dataset.replace('.tif', f'_{startdate}_{enddate}.tif')}", "w", **profile) as dst:
                dst.write(quarter_median.astype("uint16"))

        
        # now merge the 4 quarter median stacks to one stack
        # stack quarter medians
        quarter_stack = np.stack(quarter_medians, axis=3)

        # create median composite over the whole year
        year_median = calc_median(quarter_stack)

        # profile update
        profile.update(
            {
                "count": year_median.shape[2],
                "driver": "GTiff",
                "compress":"LZW",
                "predictor":2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "num_threads": 'all_cpus',
            }
        )

        # transform to rasterio's numpy axis order: bands, rows, cols
        year_median = reshape_as_raster(year_median)
        
        # write to file
        with rio.open(output_dataset, "w", **profile) as dst:
            dst.write(year_median.astype("uint16"))

 
def create_esa_worldcover(city, data_dir, reference_file, esa_url="s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"):
    """ Creates clipped esa worldcover raster WGS84 for the given Sentinel 2 image (UTM) """

    output_dataset = f"{data_dir}/esa_worldcover_{city}.tif"

    # AWS_NO_SIGN_REQUEST for ESA Worldcover access
    env = {}
    session = None

    env["AWS_NO_SIGN_REQUEST"] = 1
    env["session"] = session
        
    with rio.Env(**env):
        
        # get reference image dimensions
        with rio.open(reference_file, "r") as src:
            dst_height = src.height
            dst_width = src.width
            dst_crs = src.crs
            src_transform = src.profile.get("transform")

        dst_transform = src_transform

        with rio.open(esa_url) as src:

            vrt_options = {
                'resampling': Resampling.nearest,
                'crs': dst_crs,
                'transform': dst_transform,
                'height': dst_height,
                'width': dst_width,
            }
            with WarpedVRT(src, **vrt_options) as vrt:

                write_options = {
                    "driver": "GTiff",
                    "compress":"LZW",
                    "predictor":2,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "num_threads": 'all_cpus'
                }
                
                # Dump the aligned data into a new file.  A VRT representing
                # this transformation can also be produced by switching
                # to the VRT driver.
                rio_shutil.copy(vrt, output_dataset)
            

if __name__ == "__main__":

    geojson_file_path = "../data/osm_nominatim_Freising.geojson"

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

    from pprint import pprint
    pprint(scenes_per_year)









    