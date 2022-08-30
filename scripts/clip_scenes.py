#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-29
# authors: jsommer

""" 
    clip_scenes.py 
    
    Clip all satellite scenes to the given city boundary (GeoJSON).

"""

import json
import numpy as np
import rasterio as rio
import rasterio.features
import rasterio.mask
import rasterio.plot
import rasterio.warp
from rasterio.warp import Resampling
from matplotlib import pyplot as plt


def clip_raster(city, data_dir, composite_date, geojson_file_path, nodata=0):
    """Clips raster to the given geojson geometry """

    input_dataset = f"{data_dir}/classified/{city}_{composite_date}.tif"

    # read GeoJSON
    with open(geojson_file_path,"r") as src:
        file_content = json.load(src)

    with rio.open(input_dataset, "r") as src: 

        # transform geometry first to src's CRS
        shapes_reprojected = rasterio.warp.transform_geom(
            "EPSG:4326", # GeoJSON should be in 4326
            src.crs,
            [feature["geometry"] for feature in file_content.get("features")]
        )

        out_image, out_transform = rasterio.mask.mask(
            dataset=src,
            shapes=shapes_reprojected,
            crop=True,
            filled=True,
            nodata=nodata
        )
        profile = src.profile

    
    bands, height, width = out_image.shape

    # update profile
    profile.update(
        dict(
            transform=out_transform,
            width=width,
            height=height
        )
    )

    return profile, out_image


def save_clipped_geotiff(city, composite_date, data_dir, arry, rasterio_profile):
    """ Saves clipped GeoTiff to disk """

    output_dataset = f"{data_dir}/classified/{city}_{composite_date}_clipped.tif"

    rasterio_profile.update(dict(
        dtype="uint16",
        nodata=255,
    ))

    with rio.open(output_dataset, "w", **rasterio_profile) as dest:
        dest.write(arry)


def create_png_image(city, composite_date, data_dir, arry, rasterio_profile):
    """ Creates a PNG image for the given array and saves it to the given file_path """

    output_dataset = f"{data_dir}/classified/{city}_{composite_date}_clipped.png"

    bands = {"red": arry.copy(), "green": arry.copy(), "blue": arry.copy(), "alpha": arry.copy()}

    # 0 => 0,0,0

    # 0 => whitesmoke
    # # rgb(245, 245, 245)
    # better green: rgb(86, 192, 43)
    # green 2: rgb(73, 182, 117)
    # 1 => 0,255,0
    # better orange: rgb(249, 157, 38)
    # orange 2: rgb(255, 136, 0)
    # 3 => 255,0,0

    bands["red"][ np.where(arry == 255) ] = 0
    bands["green"][ np.where(arry == 255) ] = 0
    bands["blue"][ np.where(arry == 255) ] = 0
    bands["alpha"][ np.where(arry == 255) ] = 0

    bands["red"][ np.where(arry == 0) ] = 245
    bands["green"][ np.where(arry == 0) ] = 245
    bands["blue"][ np.where(arry == 0) ] = 245
    bands["alpha"][ np.where(arry == 0) ] = 255

    bands["red"][ np.where(arry == 1) ] = 73
    bands["green"][ np.where(arry == 1) ] = 182
    bands["blue"][ np.where(arry == 1) ] = 117
    bands["alpha"][ np.where(arry == 1) ] = 255

    bands["red"][ np.where(arry == 3) ] = 249
    bands["green"][ np.where(arry == 3) ] = 157
    bands["blue"][ np.where(arry == 3) ] = 38
    bands["alpha"][ np.where(arry == 3) ] = 255

    rgba = np.vstack([bands["red"], bands["green"], bands["blue"], bands["alpha"]])

    rasterio_profile.update(dict(
        driver="PNG",
        dtype="uint8",
        nodata=255,
        count=4
    ))

    try:
        del rasterio_profile["tiled"]
        del rasterio_profile["interleave"]
        del rasterio_profile["blockxsize"]
        del rasterio_profile["blockysize"]
        del rasterio_profile["compress"]
    except:
        pass

    with rasterio.open(f"{output_dataset}", 'w', **rasterio_profile) as dst:
        dst.write(rgba)


if __name__ == "__main__":

    data_dir = "../data"
    city = "Freising"
    geojson_file_path = f"{data}/osm_nominatim_{city}.geojson"
    years = [2017,2018,2019,2020,2021]

    for y in years:
        composite_date = y

        input_dataset = f"../data/classified/{city}_{composite_date}.tif"

        profile, clipped_array = clip_raster(
            city=city, 
            composite_date=composite_date, 
            data_dir=data_dir, 
            geojson_file_path=geojson_file_path, 
            nodata=255
        )

        save_clipped_geotiff(
            city=city, 
            composite_date=composite_date, 
            data_dir=data_dir, 
            arry=clipped_array,
            rasterio_profile=profile
        )
        
        create_png_image(
            city=city,
            composite_date=composite_date,
            data_dir=data_dir,
            arry=clipped_array,
            rasterio_profile=profile
        )

