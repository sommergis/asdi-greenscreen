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


def clip_raster(geojson_file_path, raster_file_path, nodata=0):
    """Clips raster to the given geojson geometry """

    # read GeoJSON
    with open(geojson_file_path,"r") as src:
        file_content = json.load(src)

    with rio.open(raster_file_path, "r") as src: 

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

    print(out_image.shape)
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


def save_clipped(city, composite_date, data_dir, array, rasterio_profile):
    """ Saves clipped to disk """

    output_dataset = f"{data_dir}/classified/{city}_{composite_date}_clipped.tif"

    rasterio_profile.update(dict(
        dtype="uint16",
        nodata=255,
    ))

    with rio.open(output_dataset, "w", **rasterio_profile) as dest:
        dest.write(arry)


if __name__ == "__main__":

    # city = "Freising"
    # geojson_file_path = f"../data/{city}_digitized.geojson"
    # geojson_file_path = f"../data/osm_nominatim_{city}.geojson"
    # years = [2017,2018,2019,2020,2021]

    city = "Agadir"
    geojson_file_path = f"../data/osm_nominatim_{city}.geojson"
    years = [2017,2018,2019,2020,2021]


    veg = []
    bua = []
    other = []
    for y in years:
        composite_date = y
        # input_dataset = f"../../data/composites/all_scenes_median_{city}_{composite_date}.tif"
        # output_dataset = f"../../data/composites/all_scenes_median_{city}_{composite_date}_osm_nominatim_{city}_clipped.tif"

        # input_dataset = f"../../data/classified/esa_landcover_model_MLPClassifier_2022-08-23 11:44:18.611769_score_97_3classes/{composite_date}.tif"
        # output_dataset = f"../../data/classified/esa_landcover_model_MLPClassifier_2022-08-23 11:44:18.611769_score_97_3classes/{composite_date}_osm_nominatim_{city}_clipped.tif"
        
        input_dataset = f"../data/classified/{city}_{composite_date}.tif"
        output_dataset = f"../data/classified/{city}_{composite_date}_clipped.tif"

        # with rio.open(input_dataset, "r") as src:
        #     profile = src.profile
        #     arry = src.read()

        profile, arry = clip_raster(geojson_file_path=geojson_file_path, raster_file_path=input_dataset, nodata=255)

        print(f"\nStats classified {composite_date}:\n")
        veg_area = (np.nansum((arry == 1))*100/10000) + (np.nansum((arry == 2))*100/10000)
        bua_area = np.nansum((arry == 3))*100/10000
        other_area = np.nansum((arry == 0))*100/10000
        nodata = np.nansum((arry == 255))
        
        print(f"veg: {np.nansum(arry == 1)}")
        print(f"bua: {np.nansum(arry == 3)}")
        print(f"other: {np.nansum(arry == 0)}")
        print(f"nodata: {nodata}")


        veg.append(veg_area)
        bua.append(bua_area)
        other.append(other_area)

        #dst_crs = "EPSG:3857"

        profile.update(dict(
            dtype="uint16",
            nodata=255,
        ))
        with rio.open(output_dataset, "w", **profile) as dest:
            dest.write(arry)
            # transform, width, height = rio.warp.calculate_default_transform(
            #     dest.crs, dst_crs, dest.width, dest.height, *dest.bounds
            # )

        # RGB production
        
        # print(arry[:, 500, 500])

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

        #print(rgba[:, 500, 500])

        profile.update(dict(
            driver="PNG",
            dtype="uint8",
            nodata=255,
            count=4,
            #worldfile=True,
            #crs=dst_crs,
            #transform=transform,
            #width=width,
            #height=height
        ))
        #print(profile)
        try:
            del profile["tiled"]
            del profile["interleave"]
        except:
            pass

        with rasterio.open(f"{output_dataset.replace('tif', 'png')}", 'w', **profile) as dst:
            dst.write(rgba)

            # for i in range(1, profile.get("count") +1):
            #     rio.warp.reproject(
            #         source=rgba,
            #         destination=rasterio.band(dst, i),
            #         src_transform=dst.transform,
            #         src_crs=dst.crs,
            #         dst_transform=transform,
            #         dst_crs=dst_crs,
            #         resampling=Resampling.nearest
            #     )

    print(f"Vegetation & Cropland area ha, Built up area ha, Other area ha")
    print(f"{years}")
    print(f"{veg}")
    print(f"{bua}")
    print(f"{other}")

    # # clip esa
    # input_dataset = f"../data/ESA_WorldCover_10m_2020_v100_Map_AWS_UTM32N.vrt/ESA_WorldCover_10m_2020_v100_Map_AWS_UTM32N.vrt.0.tif"
    # output_dataset = f"../data/ESA_WorldCover_10m_2020_v100_Map_AWS_UTM32N_osm_nominatim_{city}_clipped.tif"

    # profile, arry = clip_raster(geojson_file_path=geojson_file_path, raster_file_path=input_dataset)

    # with rio.open(output_dataset, "w", **profile) as dest:
    #     dest.write(arry)