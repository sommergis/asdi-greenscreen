#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-05
# authors: jsommer

""" 
    00_get_city_boundary.py 
    
    Downloads the boundary for the given city from OpenStreetMap to EPSG 3857 projection GeoJSON file.

"""

import httpx
from httpx import HTTPStatusError

import json
import rasterio as rio
import rasterio.warp

def geocode_city(city_name):
    """ Geocoding of city name to city boundary """

    # search city with Geocoding service https://nominatim.org/
    # language: english with &accept-language=en
    # country available with &addressdetails=1
    #url = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=jsonv2&accept-language=en&addressdetails=1&limit=1"

    # &polygon_geojson=1 -> city boundary
    url = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=geojson&accept-language=en&limit=1&polygon_geojson=1"

    # query service
    resp = httpx.get(url)
    if resp.status_code == 200:
        geojson = resp.json()

        return geojson

    # raise exceptions for any status != 200
    resp.raise_for_status()


def reproject_geom(geojson, src_epsg="EPSG:4326", dst_epsg="EPSG:3857"):
    """ """
    # reprojection
    shapes_reprojected = rio.warp.transform_geom(
        src_epsg,
        dst_epsg,
        [feature["geometry"] for feature in geojson["features"]]
    )
    return shapes_reprojected[0]

def save_geojson(geojson_dict, file_path):
    """ Saves geojson to given file path and reprojects it to EPSG 3857"""
    
    with open(file_path, "w") as f:
        f.write(json.dumps(geojson_dict))

    feature_col = geojson_dict
    shapes_reprojected = reproject_geom(geojson=feature_col, src_epsg="EPSG:4326", dst_epsg="EPSG:3857")

    # its only 1 feature
    feature_col["features"][0]["geometry"] = shapes_reprojected
    # add spatial reference system info (GeoJSON has natively WGS84 projection!)
    feature_col["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}

    # write geojson limited to city boundary
    with open(file_path, 'w') as f:
        f.write(json.dumps(feature_col))


if __name__ == "__main__":

    # set any city name
    city_name = "Freising"
    city_name = "Agadir"

    # geocode city name, get city boundary geojson file
    try:
        geojson_dict = geocode_city(city_name=city_name)
        
        file_path = f"../data/osm_nominatim_{city_name}.geojson"
        
        save_geojson(geojson_dict=geojson_dict, file_path=file_path)

    except HTTPStatusError as ex:
        print(ex)
    
    except Exception as ex:
        print(ex)
    

    


