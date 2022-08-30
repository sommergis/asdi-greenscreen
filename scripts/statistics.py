#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-29
# authors: jsommer

""" 
    statistics.py 
    
    Calculate statistics for vegetation or built up area pixels - also for changes to the previous year


"""

import numpy as np
import rasterio as rio
from pprint import pprint
import pandas as pd


def calc_classified_stats(input_dataset):
    """ Calculate statistics for classified input dataset """

    with rio.open(input_dataset, "r") as src:
        arry = src.read()

    veg_area = (np.nansum((arry == 1))*100/10000) + (np.nansum((arry == 2))*100/10000)
    bua_area = np.nansum((arry == 3))*100/10000
    other_area = np.nansum((arry == 0))*100/10000
    nodata = np.nansum((arry == 255))
    
    # print(f"veg: {np.nansum(arry == 1)}")
    # print(f"bua: {np.nansum(arry == 3)}")
    # print(f"other: {np.nansum(arry == 0)}")
    # print(f"nodata: {nodata}")

    return dict(
        veg_area=veg_area,
        bua_area=bua_area,
        other_area=other_area
    )

def calc_change_stats(stats, city):
    """ Calculate statistics for the statistics dict """

    years = [2017,2018,2019,2020,2021]

    df = stats
    print(df)

    change_df = pd.DataFrame(
        columns=["city", "year", "prev_year", "category", "hectares"]
    )

    rows = []
    for year in years:

        if year > 2017:
            prev_year = year - 1
        else:
            prev_year = year

        df_year = df[(df["year"] == year) & (df["city"] == city)]
        df_prev_year = df[(df["year"] == prev_year) & (df["city"] == city)]

        bua_year = df_year[df_year["category"] == "bua_area"]["hectares"].sum()
        bua_prev_year = df_prev_year[df_prev_year["category"] == "bua_area"]["hectares"].sum()
        
        veg_year = df_year[df_year["category"] == "veg_area"]["hectares"].sum()
        veg_prev_year = df_prev_year[df_prev_year["category"] == "veg_area"]["hectares"].sum()

        oth_year = df_year[df_year["category"] == "other_area"]["hectares"].sum()
        oth_prev_year = df_prev_year[df_prev_year["category"] == "other_area"]["hectares"].sum()

        delta_bua = bua_year - bua_prev_year
        delta_veg = veg_year - veg_prev_year
        
        print(f"delta_bua: {delta_bua}")
        print(f"delta_veg: {delta_veg}")

        rows.append((city, year, prev_year, "delta_bua", delta_bua, "delta_veg", delta_veg))

    change_df = change_df.from_records(rows)
    
    return change_df
    
    

if __name__ == "__main__":

    data_dir = "../data"
    city = "Freising"
    geojson_file_path = f"{data}/osm_nominatim_{city}.geojson"
    years = [2017,2018,2019,2020,2021]

    stats = {}
    change_stats = {}

    for year in years:

        if year > 2017:
            prev_year = year - 1
        else:
            continue

        composite_date = year

        input_dataset = f"{data_dir}/classified/{city}_{composite_date}_clipped.tif"
        change_dataset = f"{data_dir}/classified/{city}_{composite_date}_diff_{prev_year}_clipped.tif"
        
        stats_per_year = calc_classified_stats(input_dataset)

        stats[composite_date] = stats_per_year

        change_stats_per_year = calc_change_stats(change_dataset)

        change_stats[composite_date] = change_stats_per_year


    pprint(f"\nStats classified {stats}:\n")
    pprint(f"\nStats changes {change_stats}:\n")



