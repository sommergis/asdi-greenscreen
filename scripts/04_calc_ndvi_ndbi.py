#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-05
# authors: jsommer

""" 
    01_create_ndvi.py 
    
    Calculates NDVI

"""

import rasterio as rio
import rasterio.mask
from rasterio.warp import Resampling, aligned_target
from rasterio.features import bounds
from json import load
import numpy as np

# 1. ESA Worldcover: majority filter window for filling gaps inside urban areas (4326!)

# VRT to COGs
file = "s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt"

# geom extent
# freising UTM 32N
# bbox = [(695459.4375, 5356592), (706779.5, 5356592), (706779.5, 5369783.5), (695459.4375, 5369783.5)]

file_path = "../../data/freising_extent.geojson"
with open(file_path,"r") as src:
    file_content = load(src)
geometry = file_content["features"][0]["geometry"]

bbox = [(11.64032077789306641, 48.33063125610362931),
        (11.79197692871099434, 48.33063125610362931),
        (11.79197692871099434, 48.44903182983404122),
        (11.64032077789306641, 48.44903182983404122)]

print((bbox[1][0], bbox[3][1], bbox[1][0], bbox[3][1]))

# easier with derived bounds
bbox = bounds(geometry)

print(bbox)

resolution = 10
dst_nodata = 0


def majority_filter(array, disk_size=10):

    print(array.shape)
    #print(disk(disk_size).shape)
    #return rank.majority(array, footprint=disk(disk_size))
    return dilation(array, footprint=square(disk_size))

# read file with geom extent
with rio.open(file) as src:

    # window = rio.windows.Window.from_slices(
    #     (
    #         bbox[0],
    #         bbox[1]
    #         #bbox[1][0], #pixel_upper_left[0],
    #         #bbox[3][1] #pixel_lower_right[0]
    #     ),
    #     (
    #         bbox[2],
    #         bbox[3]
    #         #bbox[1][0], # pixel_upper_left[1],
    #         #bbox[3][1] # pixel_lower_right[1]
    #     )
    # )
    # subset = src.read(window=window)
    # print(subset)

    # # for resampling
    # out_window = rio.mask.geometry_window(src, file_content.get("features"))
    # print(out_window)

    # # resample data to target pixelsize in numpy array
    # # and use windowed read for performance
    # data = src.read(
    #     out_shape=(
    #         src.count,
    #         int(out_window.height),
    #         int(out_window.width)
    #     ),
    #     resampling=Resampling.nearest,
    #     window=out_window,
    # )

    #print(src.meta)

    # # get affine transformation by window (& new origin)
    # out_transform = src.window_transform(out_window)

    # # important: set width & height again to windowed / numpy array
    # width = data.shape[2]
    # height = data.shape[1]

    #aligned_transform = out_transform

    # # scale image transform & ensure the pixel alignment to the original pixels
    # aligned_transform, width, height = aligned_target(
    #     transform=out_transform,
    #     height=height,
    #     width=width,
    #     resolution=resolution
    # )

    # reorg geometry for rasterio masking
    geom = [feature["geometry"] for feature in file_content["features"]]

    out_image, out_transform = rio.mask.mask(
        dataset=src,  # resampled 10m windowed raster
        shapes=geom,
        crop=True,
        filled=True,
        nodata=src.nodata,
        # important for VRT / multiband files if the number of band numbers are filtered and do not match the original datasource
        #indexes=1
    )
    #print(out_image)

    # not meta - use profile for more information
    profile = src.profile

    width = out_image.shape[2]
    height = out_image.shape[1]

    profile.update({
        'count': 1,
        'driver': 'Gtiff',
        'height': height,
        'width': width,
        'transform': out_transform
    })

    print(profile)

    # apply filter
    # class 50 -> urban

    # only checkout urban
    out_image = np.where(out_image == 50, 50, 1)
    print(out_image)

    disk_size = 10
    out_image = majority_filter(np.squeeze(out_image), disk_size)
    print(out_image)

    with rio.open(f"out_filter_{disk_size}.tif", 'w', **profile) as dest:
        dest.write(out_image, 1)




