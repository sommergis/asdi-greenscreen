#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: 2022-08-29
# authors: jsommer

""" 
    classify.py 
    
    Classify all pixels of the given rasters to either vegetation or built-up area.

"""


import sys
sys.path.append(os.path.join(os.getcwd(), ".."))

import os
import json
import numpy as np
import rasterio as rio
from rasterio.plot import reshape_as_image, reshape_as_raster

from utils import read_s2_band_windowed
from remote_sensing_indices import calc_ndbi, calc_ndvi, calc_ndwi, calc_ndmi, calc_ndre1

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# plotting
import matplotlib.pyplot as plt

from joblib import dump, load
import datetime as dt
from random import randint

# GLCM features
from fast_glcm import fast_glcm_homogeneity, fast_glcm_entropy


def train(*args, esa_landcover):
    """ Builds a stack of the given arrays and trains a machine learning classifier
        using the ESA Worldcover map to vegetation & built up area pixels """

    def plot_confusion_matrix(classifier, X_test, y_test, class_names):
        """  Plot non-normalized confusion matrix """

        titles_options = [
            #("Confusion matrix, without normalization", None),
            ("Normalized confusion matrix", "true"),
        ]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                display_labels=class_names,
                cmap=plt.cm.Blues,
                normalize=normalize,
            )
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

        plt.show()


    def plot_inputdata(X, y, bands):
        """ Plot X and y data reflectance values per class """

        fig, ax = plt.subplots(1, figsize=[20,8])
        band_count = np.arange(1,bands+1)

        classes = np.unique(y)
        for class_type in classes:
            band_intensity = np.mean(X[y==class_type, :], axis=0)
            ax.plot(band_count, band_intensity, label=class_type)

        ax.set_title('Band Full Overview')
        ax.set_xlabel('Band #')
        ax.set_ylabel('Value')
        ax.legend(loc="upper right")

        plt.show()


    model_metadata = ""

    if len(args) > 1:
        # create a stack of arrays
        array_list = []

        # if squeeze is used -> use np.stack afterwards as this will introduce a new axis
        # if third axis already exists in the single arrays one can use np.dstack()
        for array in args:
            array_list.append(array.squeeze())

        stack = np.stack(array_list, axis=2)
        
    else:
        stack = args[0]

    print(stack.shape)
    rows, columns, bands = stack.shape

    stack_bands = bands

    # reshape array
    # from (1320, 1133, 12) to (1320 * 1133, 12)
    stack_reshaped = np.reshape(stack, (rows * columns, bands))

    print(stack_reshaped.shape)

    # from (1320, 1133, 1) to (1320 * 1133, 1)
    rows, columns, bands = esa_landcover.shape
    esa_landcover = np.reshape(esa_landcover, (rows * columns, bands))

    # this will get rid of that warning:
    # DataConversionWarning: A column-vector y was passed when a 1d array was expected. 
    # Please change the shape of y to (n_samples, ), for example using ravel(). y = column_or_1d(y, warn=True)
    esa_landcover = esa_landcover.squeeze()
    print(esa_landcover.shape)

    # maybe thin out landcover per class
    # esa_landcover = np.random.choice(esa_landcover, esa_landcover.shape)

    # only train built up area, vegetation and other
    # 10-40: vegetation
    # 50: bua
    # 99: other

    # keep everything from 10-50, otherwise mark as 0
    esa_landcover = np.where(
        (esa_landcover <= 60), 
        esa_landcover,
        99
    )

    # vegetation: 1
    esa_landcover = np.where(
        (esa_landcover <= 30), 
        1,
        esa_landcover
    )
    # cropland: 2
    esa_landcover = np.where(
        (esa_landcover == 40), 
        1,
        esa_landcover
    )
    # bua: 3 & # bare soil
    esa_landcover = np.where(
        (esa_landcover == 50), 
        3,
        esa_landcover
    )
    esa_landcover = np.where(
        (esa_landcover == 60), 
        3,
        esa_landcover
    )
    
    # 99 to 0
    esa_landcover = np.where(
        (esa_landcover == 99), 
        0,
        esa_landcover
    )
    
    # check for normalization
    #stack_reshaped = StandardScaler().fit_transform(stack_reshaped)

    plot_inputdata(X=stack_reshaped, y=esa_landcover, bands=stack_bands)

    # split data into training (80%), test (20%)
    X_train, X_test, y_train, y_test = train_test_split(stack_reshaped, esa_landcover, test_size=0.2, random_state=0)

    # & validation; 25% of the training set is validation; => 20% of total is validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
       
    # tree decision vars
    n_estimators = randint(5,50)

    # # instantiate classifier & fit
    # vars = dict(
    #     n_estimators = n_estimators,
    #     n_jobs = 8,
    #     random_state = 0
    # )
    # clf = RandomForestClassifier(**vars).fit(X_train, y_train)

    vars = dict(
        max_iter = 200,
        hidden_layer_sizes = (10,10,10,),
        early_stopping = True,
        activation = "relu",
        random_state=0
    )
    clf = MLPClassifier(**vars).fit(X_train, y_train)


    #
    # Accuracy metrics & model metadata
    # 
    score = clf.score(X_test, y_test)

    clf_name = str(type(clf)).strip('<').strip('>').split('.')[-1].strip("'")

    model_metadata = model_metadata + f"\nClassifier: {clf_name}"

    model_metadata = model_metadata + f"\nVars: {vars}"
    
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    model_metadata = model_metadata + f"\nConfusion matrix:\n {cm}"

    plot_confusion_matrix(clf, X_test, y_test, [])

    rep = classification_report(y_test, y_pred)
    print(rep)
    
    model_metadata = model_metadata + f"\nClassification report:\n {rep}"

    print(f"Score was: {score}")

    model_metadata = model_metadata + f"\nScore was: {score}"

    score = int(score*100)

    ##
    # feature importance
    ##
    if clf_name == "RandomForestClassifier":
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        import pandas as pd

        clf_importances = pd.Series(importances)
        
        print(clf_importances)

        model_metadata = model_metadata + f"\nFeature importances:\n{clf_importances}"

        fig, ax = plt.subplots()
        clf_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

        plt.show()

    # save ML model 
    today = dt.datetime.today()

    nb_classes = len(np.unique(esa_landcover))
    model_path = f"esa_landcover_model_{clf_name}_{today}_score_{score}_{nb_classes}classes.joblib"
    dump(clf, model_path)

    with open(f"{os.path.splitext(model_path)[0]}.txt", "w") as meta:
        meta.write(model_metadata)

    return model_path


def predict(*args, model_path=""):
    """ Predict the Landcover class using the given model for Sentinel2 12 band arrays """

    if len(args) > 1:
        # create a stack of input arrays
        array_list = []
        for array in args:
            array_list.append(array.squeeze())

        #stack = np.dstack(array_list)
        
        # Dimos tip
        stack = np.stack(array_list, axis=2)
        print(stack.shape)
        # axis=2
        rows, columns, bands = stack.shape

    else:
        stack = args[0]
    
    rows, columns, bands = stack.shape

    print(f"stack.shape: {stack.shape}")
    
    # reshape array
    stack_reshaped = np.reshape(stack, (rows * columns, bands))

    clf = load(model_path)

    # scaling for prediction
    #stack_reshaped = StandardScaler().fit_transform(stack_reshaped)

    result = clf.predict(stack_reshaped)
    result_proba = clf.predict_proba(stack_reshaped)

    # # single band output - reshaping
    print(f"result.shape: {result.shape}")
    bands = 1
    result = np.reshape(result, (bands, rows, columns))

    print(f"result.shape: {result.shape}")

    # # single band output - reshaping
    print(f"result_proba.shape: {result_proba.shape}")
    bands = result_proba.shape[1] # probability per class!

    result_probas = {}
    for n in range(0, bands):

        class_proba = result_proba[:, n]
        class_proba = class_proba.reshape((rows, columns))
        
        result_probas[n] = class_proba

        print(f"result_proba.shape: {class_proba.shape}")


    return result, result_probas


def prepare_features(allbands, esa_landcover):
    """ Prepare data for ML training """

    # rasterio numpy axis order
    # bands, rows, columns = stack.shape
    bands, rows, cols = allbands.shape

    # extract bands from stack
    b01 = allbands[0, :, :]
    b02 = allbands[1, :, :]
    b03 = allbands[2, :, :]
    b04 = allbands[3, :, :]
    b05 = allbands[4, :, :]
    b06 = allbands[5, :, :]
    b07 = allbands[6, :, :]
    b08 = allbands[7, :, :]
    b8A = allbands[8, :, :]
    b09 = allbands[9, :, :]
    b11 = allbands[10, :, :]
    b12 = allbands[11, :, :]

    #
    # compute remote sensing indices
    #

    ndvi = calc_ndvi(red_array=b04, nir_array=b08)
    ndbi = calc_ndbi(swir_array=b11, nir_array=b08)
    ndwi = calc_ndwi(green_array=b03, nir_array=b08)
    ndmi = calc_ndmi(swir_array=b11, nir_array=b08)
    ndre1 = calc_ndre1(rededge_array_1=b05, rededge_array_2=b06)

    # swap axis for image format for ML classifier
    allbands = reshape_as_image(allbands)

    #
    # extract GLCM textures
    #

    # strech normalized NDVI values (-1.0 to 1.0) to 0-255 RGB values
    glcm_input = (ndvi*255).astype('uint8')
    
    homogeneity = fast_glcm_homogeneity(glcm_input)
    entropy = fast_glcm_entropy(glcm_input)

    # stack all features together
    feature_stack = np.dstack([ndvi, ndwi, ndre1, homogeneity, entropy])
    
    # swap axis for image format for ML classifier
    esa_landcover = reshape_as_image(esa_landcover)

    # check for x/y dimensions - z dimension is different
    assert [feature_stack.shape[0],feature_stack.shape[1]] == [esa_landcover.shape[0],esa_landcover.shape[1]], \
        f"S2 median stack and esa worldcover do not have the same dimensions!"

    # take care of nans for ML classifier
    # for reflectances NaNs -> 0
    feature_stack = np.nan_to_num(feature_stack, nan=-99999)

    return feature_stack, esa_landcover


def save_feature_stack(city, composite_date, data_dir, all_bands, rasterio_profile):
    """ Saves feature stack to disk """

    # rasterio profile for saving training data to disk
    rasterio_profile.update({
        "dtype": "float32",
        "count": allbands.shape[2],
        "nodata": -99999
    })

    # reshape to format for rasterio (bands, rows, cols, bands)
    allbands_img = reshape_as_raster(allbands)

    # for testing only - write band to file system
    with rio.open(f"{data_dir}/features/{city}_{composite_date}.tif", "w", **rasterio_profile) as dest:
        dest.write(allbands_img)


def save_classified(city, composite_date, data_dir, classified, classified_probas, rasterio_profile):
    """ Saves classified and classified probability GeoTiffs to disk """ 

    # write band to disk
    with rio.open(f"{data_dir}/classified/{city}_{composite_date}.tif", "w", **rasterio_profile) as dest:
        dest.write(classified)

    rasterio_profile.update({
        "dtype": np.float32,
        "count": 1,
        "nodata": -1
    })

    # write probability predictions to disk
    for cl in classified_probas.keys():
        class_array = classified_probas.get(cl)
        
        with rio.open(f"{data_dir}/classified/{city}_{composite_date}_{cl}_proba.tif", "w", **rasterio_profile) as dest:
            dest.write(class_array, 1)


if __name__ == '__main__':

    # ML training (True) or prediction (False)
    training = False

    city = "Freising"    
    composite_date = "2020"
    year = composite_date

    data_dir = f"../data/"

    # create output directories if not present
    dirs = ["features", "classified"]
    for dir in dirs:
        if not os.path.exists(f"{data_dir}/{dir}"):
            os.mkdir(f"{data_dir}/{dir}")

    geojson_file_path = f"{data_dir}/osm_nominatim_{city}.geojson"
    input_dataset = f"{data_dir}/composites/median_{city}_{composite_date}.tif"
    training_dataset = f"{data_dir}/esa_worldcover_{city}.tif"

    model_path = "./esa_landcover_model_MLPClassifier_2022-08-26 10:47:46.578398_score_88_3classes.joblib"

    #
    # read input GeoTiffs
    #
    with rio.open(input_dataset, "r") as src:
        allbands = src.read()
        profile = src.profile

    print(f"Composite shape: {allbands.shape}")

    with rio.open(training_dataset) as src:
        esa_landcover = src.read()
        profile = src.profile

    print(f"ESA worldcover shape: {esa_landcover.shape}")

    # 
    # prepare features
    #
    feature_stack, esa_landcover = prepare_features(
        allbands=allbands,
        esa_landcover=esa_landcover
    )
    
    #
    # save feature stack to disk
    #
    print(f"All bands shape: {feature_stack.shape}")

    save_feature_stack(
        city=city,
        composite_date=composite_date,
        feature_stack=feature_stack,
        rasterio_profile=profile
    )

    if training == True:

        model_path = train(feature_stack, esa_landcover=esa_landcover)

        # add training dataset name to model meta
        model_meta_path = f"{os.path.splitext(model_path)[0]}.txt"
        
        with open(model_meta_path, 'a') as fp:
            fp.write(f"\n\ntraining dataset(s): {input_dataset}")

        print(model_path)

    elif training == False:
        
        # prediction
        classified, classified_probas = predict(
            feature_stack, 
            model_path=model_path
        )

        profile.update({
            "dtype": np.uint8,
            "count": classified.shape[0],
            "nodata": 255
        })

        #
        # save prediction to disk
        # 
        save_classified(
            city=city, 
            composite_date=composite_date, 
            data_dir=data_dir, 
            classified=classified, 
            classified_probas=classified_probas, 
            rasterio_profile=profile
        )