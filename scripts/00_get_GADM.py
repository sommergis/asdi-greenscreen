#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-05
# authors: jsommer

""" 
    00_get_GADM.py 
    
    Downloads the GADM database for the given city and extracts the GADM Level 3 as layer to EPSG 3857 projection.

"""

import httpx
# from 2 letter country codes (OSM) to 3 letter codes for GADM
from iso3166 import countries
import zipfile

def geocode_city(city_name):
    """ Geocoding of city name to country """

    # search city with Geocoding service https://nominatim.org/
    # language: english with &accept-language=en
    # country available with &addressdetails=1
    url = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=jsonv2&accept-language=en&addressdetails=1&limit=1"

    # query service
    resp = httpx.get(url)
    if resp.status_code == 200:
        geocoder_resp = resp.json()
        # get first result - should be only one
        address = geocoder_resp[0].get("address")
        country_code_2_letter = address.get("country_code")

        country = countries.get(country_code_2_letter)
        country_code_3_letter = country.alpha3

        return country_code_3_letter

def get_download_url_gadm(country_code_3_letter, level=3):
    """ Returns the download URL (geojson) for the given country code at the given level """

    # download GADM for country
    # e.g. https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_DEU_3.json.zip
    level = 3
    gadm_url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code_3_letter.upper()}_{level}.json.zip"

    return gadm_url

def download_gadm(url, file_path):
    """ Downloads GADM url & extracts zip"""

    resp = httpx.get(url)
    if resp.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(resp.content)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_path.split(".zip")[0])

if __name__ == "__main__":

    city_name = "Freising"
    gadm_level = 3

    # geocode city name, get country code
    country_code_3_letter = geocode_city(city_name=city_name)

    # build download URL for GADM Level 3
    download_url = get_download_url_gadm(country_code_3_letter=country_code_3_letter, level=gadm_level)

    # download & extract geojson
    download_gadm(url=download_url, file_path=f"../data/{city_name}_gadm_level_{gadm_level}.zip")


