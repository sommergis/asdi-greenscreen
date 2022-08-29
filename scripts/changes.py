#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-29
# authors: jsommer

""" 
    changes.py 
    
    Change detection of classified urban green areas and built up areas over time.
    Creates GeoTiff and PNG images.

"""

import numpy as np
import rasterio as rio
import rasterio.features
import rasterio.mask
import rasterio.plot
import rasterio.warp
from rasterio.warp import Resampling
from matplotlib import pyplot as plt


def calc_diff(arry1, arry2):
    """ Calculates difference for arry1 - arry2 """

    return arry1 - arry2


def create_change_geotiff(input_dataset_cur_year, input_dataset_prev_year):
    """ Creates a change GeoTiff image from the given numpy arrays and returns the difference numpy array """

    with rio.open(input_dataset_cur_year, "r") as src:
        arry1 = src.read()
        profile = src.profile

    with rio.open(input_dataset_prev_year, "r") as src:
        arry2 = src.read()

    arry1 = np.where(arry1 == 255, np.nan, arry1).astype("float32")
    arry2 = np.where(arry2 == 255, np.nan, arry2).astype("float32")

    # calc difference for change detection
    diff_arry = (arry1 - arry2)

    profile.update(dict(
        dtype = "float32"
    ))

    # write as Geotiff
    with rio.open(output_dataset, "w", **profile) as dst:
        dst.write(diff_arry)

    return diff_arry


def create_change_png_images(diff_arry, file_path):
    """ Creates a PNG image from the given file_path """

    arry = diff_arry.copy()
    bands = {"red": arry.copy(), "green": arry.copy(), "blue": arry.copy(), "alpha": arry.copy()}

    # print(np.unique(arry))
    # print(f"nb of -3s: {np.count_nonzero(arry[np.where(arry == -3)])}")
    # print(f"nb of -2s: {np.count_nonzero(arry[np.where(arry == -2)])}")
    # print(f"nb of -1s: {np.count_nonzero(arry[np.where(arry == -1)])}")
    # print(f"nb of 1s: {np.count_nonzero(arry[np.where(arry == 1)])}")
    # print(f"nb of 2s: {np.count_nonzero(arry[np.where(arry == 2)])}")
    # print(f"nb of 3s: {np.count_nonzero(arry[np.where(arry == 3)])}")
    # print(f"nb of nans: {np.count_nonzero(np.isnan(arry))}")
    # print(f"nb of array: {arry.size}")
    # print(f"nb of 0s: {arry.size - np.count_nonzero(arry)}")

    #
    # Colors
    #

    # -2 => 101, 156, 60 (green)
    # 0 => 254, 254, 224 (yellow)
    # 2 => 102, 51, 1 (brown)

    bands["red"][ np.where(arry == 255) ] = 0
    bands["green"][ np.where(arry == 255) ] = 0
    bands["blue"][ np.where(arry == 255) ] = 0
    bands["alpha"][ np.where(arry == 255) ] = 0

    bands["red"][ np.where(arry == np.nan) ] = 0
    bands["green"][ np.where(arry == np.nan) ] = 0
    bands["blue"][ np.where(arry == np.nan) ] = 0
    bands["alpha"][ np.where(arry == np.nan) ] = 0

    bands["red"][ np.where(arry <= -2) ] = 101
    bands["green"][ np.where(arry <= -2) ] = 156
    bands["blue"][ np.where(arry <= -2) ] = 60
    bands["alpha"][ np.where(arry <= -2) ] = 255

    bands["red"][ np.where(arry == 0) ] = 254
    bands["green"][ np.where(arry == 0) ] = 254
    bands["blue"][ np.where(arry == 0) ] = 224
    bands["alpha"][ np.where(arry == 0) ] = 255

    bands["red"][ np.where(arry >= 2) ] = 102
    bands["green"][ np.where(arry >= 2) ] = 51
    bands["blue"][ np.where(arry >= 2) ] = 1
    bands["alpha"][ np.where(arry >= 2) ] = 255

    # Stack RGBA image
    rgba = np.vstack([bands["red"], bands["green"], bands["blue"], bands["alpha"]])
    rgba = np.nan_to_num(rgba, 0)

    profile.update(
        dict(
            driver="PNG",
            dtype="uint8",
            count=4,
            nodata=255
            #worldfile=True,
        )
    )

    # remove GeoTiff profile items
    try:
        del profile["tiled"]
        del profile["interleave"]
    except:
        pass

    with rasterio.open(f"{file_path}", 'w', **profile) as dst:
        dst.write(rgba)


if __name__ == "__main__":
    
    city = "Agadir"
    years = [2017,2018,2019,2020,2021]

    for year in years:

        if year > 2017:
            prev_year = year - 1
        else:
            continue

        input_dataset_cur_year = f"../data/classified/{city}_{year}_clipped.tif"
        input_dataset_prev_year = f"../data/classified/{city}_{prev_year}_clipped.tif"
        
        output_dataset = f"../data/classified/{city}_clipped_{year}_diff_{prev_year}.tif"


        diff_arry = create_change_geotiff(
            input_dataset_cur_year=input_dataset_cur_year,
            input_dataset_prev_year=input_dataset_prev_year
        )

        create_change_png_images(
            diff_arry, 
            file_path
        )