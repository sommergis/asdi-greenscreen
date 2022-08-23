#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: <YYYY-MM-DD>
# authors: <author>

""" 
    01_get_satellite_scenes.py 
    
    Get all satellite scenes for the last 5 years for the given city boundary (GeoJSON) and a certain date:
    e.g.: 2017-06-15, 2018-06-15, 2019-06-15, 2020-06-15, 2021-06-15, 2022-06-15

"""

from satsearch import Search
import datetime as dt

import rasterio as rio
from rasterio.features import bounds
import json

def get_scenes(startdate, enddate, geojson, max_cloudcover=20):
    """ """

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