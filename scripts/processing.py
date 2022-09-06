#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-29
# authors: jsommer

""" 
    processing.py 
    
    Main processing script for asdi greenscreen project with the following steps:
    - 00 get city boundary
    - 01 get satellite scenes and create median composites, mask clouds in composites
    - 02 get and reproject ESA worldcover to UTM
    - 03 ML feature engineering and training for 2020
    - 04 ML prediction for 2017, 2018, 2019, 2021
    - 05 clip classified to geometry
    - 06 create PNG images for classified results
    - 07 calculate changes and create PNG images for changes
    - 08 calculate statistics

"""

import os, json
from httpx import HTTPStatusError
import rasterio as rio

from get_city_boundary import geocode_city, save_geojson
from get_satellite_scenes import create_median_composites, create_esa_worldcover
from utils import get_geojson_center, get_epsg_utm_zone
from remote_sensing_indices import calc_ndvi, calc_ndwi, calc_ndre1
from classify import prepare_features, save_feature_stack, train, predict, save_classified
from clip_scenes import clip_raster, save_clipped_geotiff, create_png_image
from changes import create_change_geotiff,  create_change_png_image
from statistics import calc_classified_stats, calc_change_stats

import pandas as pd

if __name__ == "__main__":

    data_dir = "../data"

    #city = "Agadir"
    city = "Freising"
    years = [2017,2018,2019,2020,2021]
    geojson_file_path = f"{data_dir}/osm_nominatim_{city}.geojson"


    print(f"Checking for output directories..")
    try:
        # create output directories if not present
        dirs = ["composites", "features", "classified"]
        for dir in dirs:
            if not os.path.exists(f"{data_dir}/{dir}"):
                os.mkdir(f"{data_dir}/{dir}")

    except Exception as ex:
        print("Could not create output directories!")
        print(ex)


    # 00 get city boundary
    print(f"Getting city geometry for {city}..")
    try:
        geojson_dict = geocode_city(city_name=city)
        
        save_geojson(geojson_dict=geojson_dict, file_path=geojson_file_path)

    except HTTPStatusError as ex:
        print(ex)
    
    except Exception as ex:
        print(ex)


    try:
        # 01 get satellite scenes and create median composites, mask clouds in composites
        print(f"Getting satellite scenes and creating median composites for each year for {city}..")
        print(f"Hint: this may take a while and is dependent on your internet connection!")

        create_median_composites(
            geojson_file_path=geojson_file_path,
            city=city, 
            data_dir=data_dir,
            years=years, 
            # ok for city of Freising, more sunny regions (or bigger regions may require a lower value in order to process fewer scenes)
            max_cloudcover=40
        )

        # 02 get and reproject ESA worldcover to UTM
        print(f"Creating ESA worldcover map for {city}..")
        
        create_esa_worldcover(
            city=city,
            data_dir=data_dir,
            reference_file=f"{data_dir}/composites/median_{city}_2020.tif"
        )

        # 03 ML feature engineering and training for 2020
        
        print(f"Starting Machine Learning training for {city} in 2020..")

        training_date = 2020
        composite_dataset = f"{data_dir}/composites/median_{city}_{training_date}.tif"
        training_dataset = f"{data_dir}/esa_worldcover_{city}.tif"

        assert os.path.exists(composite_dataset) and os.path.exists(training_dataset), \
            f"Median composite {composite_dataset} or ESA Worldcover {training_dataset} do not exist!"

        #
        # read input GeoTiffs
        #
        with rio.open(composite_dataset, "r") as src:
            allbands = src.read()
            profile = src.profile

        print(f"Composite shape: {allbands.shape}")

        with rio.open(training_dataset) as src:
            esa_landcover = src.read()
            profile = src.profile

        print(f"ESA worldcover shape: {esa_landcover.shape}")

        print(f"Preparing feature stack for {city}..")

        # prepare feature stack
        feature_stack, esa_landcover = prepare_features(
            allbands=allbands,
            esa_landcover=esa_landcover
        )

        print(f"Saving feature stack for {city}..")
        save_feature_stack(
            city=city, 
            composite_date=training_date, 
            data_dir=data_dir, 
            feature_stack=feature_stack, 
            rasterio_profile=profile
        )

        # # training
        # print(f"Training..")
        # model_path = train(feature_stack, esa_landcover=esa_landcover)

        # # add training dataset name to model meta
        # model_meta_path = f"{os.path.splitext(model_path)[0]}.txt"
        
        # with open(model_meta_path, 'a') as fp:
        #     fp.write(f"\n\composite dataset(s): {composite_dataset}")
        #     fp.write(f"\n\ntraining dataset(s): {training_dataset}")

        model_path = "esa_landcover_model_MLPClassifier_2022-08-29 13:04:16.912490_score_96_3classes.joblib"

        print(f"ML model path: {model_path}")

        # 04 ML prediction for all years (incl. 2020)
        for year in years:

            print(f"Machine learning prediction for {city} in {year}..")

            composite_date = year
            composite_dataset = f"{data_dir}/composites/median_{city}_{composite_date}.tif"

            assert os.path.exists(composite_dataset), f"Median composite {composite_dataset} does not exist!"

            #
            # read input GeoTiffs
            #
            with rio.open(composite_dataset, "r") as src:
                allbands = src.read()
                profile = src.profile

            print(f"Composite shape: {allbands.shape}")
            print(f"ESA worldcover shape: {esa_landcover.shape}")

            # prepare feature stack per year
            feature_stack, esa_landcover = prepare_features(
                allbands=allbands,
                esa_landcover=esa_landcover
            )

            # prediction
            classified, classified_probas = predict(
                feature_stack, 
                model_path=model_path
            )

            profile.update({
                "dtype": "uint8",
                "count": classified.shape[0],
                "nodata": 255
            })

            # save prediction to disk
            print(f"Saving classification for {city} in {year}..")
            save_classified(
                city=city, 
                composite_date=composite_date, 
                data_dir=data_dir, 
                classified=classified, 
                classified_probas=classified_probas, 
                rasterio_profile=profile
            )
        
            # 05 clip classified to geometry
            print(f"Clipping classification for {city} in {year} to city boundary..")

            profile, clipped_array = clip_raster(
                city=city, 
                composite_date=composite_date, 
                data_dir=data_dir,
                geojson_file_path=geojson_file_path,
                nodata=255
            )

            # save clipped to disk
            save_clipped_geotiff(
                city=city, 
                composite_date=composite_date, 
                data_dir=data_dir, 
                arry=clipped_array,
                rasterio_profile=profile
            )
        
            # 06 create PNG images for classified results
            print(f"Creating PNG images for classification for {city} in {year}..")
            create_png_image(
                city=city,
                composite_date=composite_date,
                data_dir=data_dir,
                arry=clipped_array,
                rasterio_profile=profile
            )

        # 07 calculate changes after everything is ready
        for year in years:
            
            print(f"Computing change detection for classification for {city} in {year}..")
            
            if year > 2017:
                prev_year = year - 1
            else:
                continue

            diff_arry, profile = create_change_geotiff(
                city=city, 
                baseline_year=year, 
                compare_year=prev_year,
                data_dir=data_dir
            )

            create_change_png_image(
                city=city,
                baseline_year=year,
                compare_year=prev_year,
                data_dir=data_dir,
                diff_arry=diff_arry,
                rasterio_profile=profile
            )

        stats = []
        change_stats = []
        for year in years:
            # 08 calculate statistics for each
            print(f"Computing stats for {city} in {year}..")

            input_dataset = f"{data_dir}/classified/{city}_{year}_clipped.tif"
            change_dataset = f"{data_dir}/classified/{city}_{year}_diff_{prev_year}_clipped.tif"

            row = dict(
                city=city,
                year=year
            )
            row.update(
                calc_classified_stats(
                    input_dataset=input_dataset
                )
            )
            stats.append(row)

        df = pd.DataFrame()
        df = df.from_records(stats)
        print(df)

        # save to json
        stats_file_path = f"{data_dir}/classified/{city}_clipped_stats.json"
        with open(stats_file_path, "w") as out:
            out.write(json.dumps(stats))

        # convert from wide to long format
        df = df.melt(['city', 'year'], var_name="category", value_name='hectares')

        # save to csv
        stats_file_path = f"{data_dir}/classified/{city}_clipped_stats.csv"
        df.to_csv(stats_file_path, index=False)

        df = calc_change_stats(
            city=city,
            stats=df
        )

        # save to json
        stats_file_path = f"{data_dir}/classified/{city}_clipped_stats_change.json"
        df.to_json(stats_file_path, index=False, orient="table")

        # save to csv
        stats_file_path = f"{data_dir}/classified/{city}_clipped_stats_change.csv"
        df.to_csv(stats_file_path, index=False)

    except Exception as ex:
        print(ex)