#!/usr/bin/env python
# -*- coding: utf-8 -*-

# last update: <YYYY-MM-DD>
# authors: <author>

""" 
    05_classify.py 
    
    Classify all pixels of the given rasters to either vegetation or built-up area.

"""

import os
import json
import numpy as np
import rasterio as rio
from rasterio.plot import reshape_as_image, reshape_as_raster

import sys
sys.path.append(os.path.join(os.getcwd(), ".."))

from utils import read_s2_band_windowed
from calc_ndvi_ndbi import calc_ndbi, calc_ndvi, calc_ndwi, calc_ndmi, calc_ndre1

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# plotting
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.model_selection import train_test_split
from joblib import dump, load
import datetime as dt
from random import randint


def reclass_array(ndvi_array, ndbi_array):
    """ Reclassify the given ndvi and ndbi arrays to vegetation and built up area pixels """

    # vegetation: 1
    veg = np.where(
        (ndvi_array >= 0.4) & (ndbi_array <= -0.05), 
        1,
        0
    )
    # built-up area: 2
    bua = np.where(
        (ndvi_array < 0.4) & (ndbi_array > -0.05),
        2,
        0
    )
    # combination of both
    return veg + bua


def cluster_arrays(*args):
    """ Creates a stack of the given arrays and clusters them to vegetation and built up area pixels """
    
    if len(args) > 0:
        array_list = []
        # create a stack of arrays
        for array in args:
            array_list.append(array.squeeze())

        # Dimos tip
        stack = np.stack(array_list, axis=2)

    else:
        stack = args[0]

    print(stack.shape)

    rows, columns, bands = stack.shape

    # reshape array
    #stack_reshaped = stack.reshape((-1,1))
    stack_reshaped = np.reshape(stack, (rows * columns, bands))
    #ndvi_array_reshaped = ndvi_array.reshape((-1,1))

    print(stack_reshaped.shape)

    # instantiate classifier & fit
    #k_means = KMeans(n_clusters=2, random_state=0).fit(ndvi_array_reshaped)
    #k_means = MiniBatchKMeans(n_clusters=6).fit(stack_reshaped)
    clf = DBSCAN().fit(stack_reshaped)

    cluster = clf.labels_

    # reshape again
    #labels = cluster.reshape(ndvi_array.shape)    
    #labels = cluster.reshape(stack.shape)
    
    # single band output
    bands = 1
    labels = np.reshape(cluster, (bands, rows, columns))

    print(labels.shape)


    return labels


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

        # this will yield better results, but seems wrong!
        # stack = np.dstack(array_list)
        # rows, columns, bands = stack.shape
        # stack = stack.reshape((bands, rows, columns))
        
        # seems correct, but results are not good!
        #stack = np.stack(array_list, axis=0)

        # Dimos tip
        stack = np.stack(array_list, axis=2)
        print(stack.shape)
        # axis=2
        rows, columns, bands = stack.shape
        #from rasterio.plot import reshape_as_raster
        #stack = reshape_as_raster(stack)

        # # axis=0
        # bands, rows, columns = stack.shape
        # print(stack.shape)

        #import sys
        #sys.exit()

    else:
        stack = args[0]

    print(stack.shape)

    # rasterio numpy axis order
    # bands, rows, columns = stack.shape
    
    rows, columns, bands = stack.shape

    stack_bands = bands

    # should look like (1320, 1133, 10)
    print(stack.shape)

    # reshape array
    # from (12, 1320, 1133) to (1320 * 1133, 12)
    stack_reshaped = np.reshape(stack, (rows * columns, bands))

    print(stack_reshaped.shape)

    # from (1, 1320, 1133) to (1320 * 1133, 1)
    bands, rows, columns = esa_landcover.shape
    esa_landcover = reshape_as_image(esa_landcover)
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

    # # KNeighborsClassifier only
    # n_neighbors = randint(2, 10)
    # print(f"n_neighbors: {n_neighbors}")

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

    # vars = dict(
    #     probability=True, 
    #     max_iter=100
    # )
    # clf = SVC(**vars).fit(X_train, y_train)

    # vars = dict(
    #     n_estimators=n_estimators,
    # )
    # clf = GradientBoostingClassifier(**vars).fit(X_train, y_train)

    #clf = GaussianNB().fit(X_train, y_train)

    # not applicable for memory reasons
    # vars = dict(
    #     kernel = 1.0 * RBF(1.0)
    # )
    # clf = GaussianProcessClassifier(**vars).fit(X_train, y_train)

    # the higher the nb of neighbors the higher the score (but also runtime!)
    # vars = dict(
    #     n_neighbors=7, 
    #     n_jobs=8
    # )
    # clf = KNeighborsClassifier(**vars).fit(X_train, y_train)

    # vars = dict(
    #     n_jobs=8, max_iter=200, multi_class="multinomial"
    # )
    # clf = LogisticRegression(**vars).fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    clf_name = str(type(clf)).strip('<').strip('>').split('.')[-1].strip("'")

    model_metadata = model_metadata + f"\nClassifier: {clf_name}"

    model_metadata = model_metadata + f"\nVars: {vars}"
    
    # metrics: confusion matrix, plotting, score
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

        #feature_names = []
        clf_importances = pd.Series(importances)
        
        print(clf_importances)

        model_metadata = model_metadata + f"\nFeature importances:\n{clf_importances}"

        fig, ax = plt.subplots()
        clf_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

        plt.show()

    # save model 
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


def create_median_stack(year, city):
    """ Creates a median stack for the given year """

    def nanmedian(all_scenes_stack):
        """ Efficient median computation with cython """
        import bottleneck as bn

        return bn.nanmedian(
          np.where(all_scenes_stack < 1, np.nan, all_scenes_stack).astype("float32"), 
          axis=3
        ).astype("uint16")
    
    def nanmedian_np(all_scenes_stack):
        """ Regular median computation with numpy """
        return np.nanmedian(
          np.where(all_scenes_stack < 1, np.nan, all_scenes_stack).astype("float32"), 
          axis=3
        ).astype("uint16")

    dates = [
        f"{year}-01-01_{year}-03-31", 
        f"{year}-04-01_{year}-06-30",
        f"{year}-07-01_{year}-09-30",
        f"{year}-10-01_{year}-12-31"
    ]
    from glob import glob

    all_scenes = []
    for composite_date in dates:
        input_dataset = f"../../data/composites/all_scenes_median_{city}_{composite_date}_*.tif"
        print(input_dataset)
        f = glob(input_dataset)[0]
        print(f)

        with rio.open(f, "r") as src:
            scene = src.read()
            all_scenes.append(scene)
            profile = src.profile

    # stack
    all_scenes_stack = np.stack(all_scenes, axis=3)

    print(all_scenes_stack.shape)
    all_scenes_median = nanmedian(all_scenes_stack)
    
    #.reshape((bands, rows, cols, time))
    # all_scenes_median_1 = median(all_scenes_stack) 
    # all_scenes_median_2 = npmedian(all_scenes_stack) 

    # assert all_scenes_median_1.all() == all_scenes_median_2.all()

    # # create median - watch out for zeros!
    # print((all_scenes_stack < 1).sum())
    # print(np.count_nonzero(np.isnan(all_scenes_stack)))

    # all_scenes_stack = np.where(all_scenes_stack < 1, np.nan, all_scenes_stack).astype("float32")
    # print((all_scenes_stack < 1).sum())
    # print(np.count_nonzero(np.isnan(all_scenes_stack)))

    # # 0 will be bad for median calculation!

    # # show(all_scenes_stack[:, :, 4, 4])
    # all_scenes_median = np.nanmedian(all_scenes_stack, axis=3).astype("uint16")

    #all_scenes_median = reshape_as_raster(all_scenes_median)
    print(all_scenes_median.shape)

    with rio.open(f"../../data/composites/all_scenes_median_{city}_{year}.tif", "w", **profile) as dest:
        dest.write(all_scenes_median.astype("uint16"))



if __name__ == '__main__':

    training = False

    city = "Freising"
    geojson_file_path = f"../../data/osm_nominatim_{city}.geojson"
    composite_date = "2021"
    input_dataset = f"../../../data/composites/all_scenes_median_{city}_{composite_date}_osm_nominatim_{city}_clipped.tif"
    year = composite_date

    #model_path = "./esa_landcover_model_MLPClassifier_2022-08-21 13:50:47.956793_score_88_4classes.joblib"
    #model_path = "./esa_landcover_model_MLPClassifier_2022-08-21 13:59:26.745505_score_96_3classes.joblib"
    #model_path = "./esa_landcover_model_RandomForestClassifier_2022-08-21 14:32:09.999005_score_91_4classes.joblib"
    #model_path = "./esa_landcover_model_RandomForestClassifier_2022-08-21 14:30:43.437518_score_96_3classes.joblib"
    #model_path = "./esa_landcover_model_RandomForestClassifier_2022-08-21 21:39:56.428012_score_96_3classes.joblib"
    #model_path = "./esa_landcover_model_RandomForestClassifier_2022-08-21 21:57:40.078347_score_96_3classes.joblib"
    #model_path = "./esa_landcover_model_RandomForestClassifier_2022-08-21 22:26:17.055292_score_95_3classes.joblib"
    
    # more pixely
    #model_path = "./esa_landcover_model_RandomForestClassifier_2022-08-21 22:32:53.245841_score_96_3classes.joblib"
    
    # more compact bua
    #model_path = "./esa_landcover_model_MLPClassifier_2022-08-21 22:41:43.372977_score_96_3classes.joblib"
    #model_path = "./esa_landcover_model_MLPClassifier_2022-08-22 21:15:38.150452_score_97_3classes.joblib"
    
    # now with 0,2,3 as classes
    model_path = "./esa_landcover_model_MLPClassifier_2022-08-23 11:44:18.611769_score_97_3classes.joblib"

    # create_median_stack(year=year, city=city)
    # import sys
    # sys.exit()

    with rio.open(input_dataset, "r") as src:
        allbands = src.read()
        profile = src.profile

    # rasterio order
    bands, rows, cols = allbands.shape

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

    #esa_landcover, profile = read_s2_band_windowed(s2_file_path="../../data/ESA_WorldCover_10m_2020_v100_Map_AWS_UTM32N.vrt/ESA_WorldCover_10m_2020_v100_Map_AWS_UTM32N.vrt.0.tif", geojson_file_path=geojson_file_path)

    with rio.open(f"../../data/ESA_WorldCover_10m_2020_v100_Map_AWS_UTM32N_osm_nominatim_{city}_clipped.tif", "r") as src:
        esa_landcover = src.read()
        profile = src.profile

    # assert b01.shape == b02.shape == b03.shape == b04.shape == b05.shape \
    #     == b06.shape == b07.shape == b08.shape == b8A.shape == b09.shape \
    #     == b11.shape == b12.shape == esa_landcover.shape, \
    #         f"""shapes of bands differ: {b01.shape}, {b02.shape}, {b03.shape}, {b04.shape}, {b05.shape}, 
    #             {b06.shape}, {b07.shape}, {b08.shape}, {b8A.shape}, {b09.shape}, {b11.shape},
    #             {b12.shape}, {esa_landcover.shape} """
    
    ndvi = calc_ndvi(red_array=b04, nir_array=b08)
    ndbi = calc_ndbi(swir_array=b11, nir_array=b08)
    ndwi = calc_ndwi(green_array=b03, nir_array=b08)
    ndmi = calc_ndmi(swir_array=b11, nir_array=b08)
    ndre1 = calc_ndre1(rededge_array_1=b05, rededge_array_2=b06)



    # swap axis for image format
    allbands = reshape_as_image(allbands)

    ##
    # extract GLCM textures
    ##
    from fast_glcm import fast_glcm_homogeneity, fast_glcm_contrast, fast_glcm_mean, fast_glcm_entropy, fast_glcm_dissimilarity

    # normalize first for reflectances
    #glcm_input = (((b08 - b08.min()) / (b08.max() - b08.min()) )*255).astype('uint8')
    
    # # good with homogeneity, glcm_mean, entropy
    # glcm_input = (((b8A - b8A.min()) / (b8A.max() - b8A.min()) )*255).astype('uint8')
    
    # # good with homogeneity, glcm_mean, entropy
    #glcm_input = (((b03 - b03.min()) / (b03.max() - b03.min()) )*255).astype('uint8')

    # good with homogeneity, glcm_mean, entropy
    #glcm_input = (((b04 - b04.min()) / (b04.max() - b04.min()) )*255).astype('uint8')

    #glcm_input = (((b02 - b02.min()) / (b02.max() - b02.min()) )*255).astype('uint8')
    #glcm_input = (ndwi*255).astype('uint8')
    #glcm_input = (ndbi*255).astype('uint8')


    glcm_input = (ndvi*255).astype('uint8')
    
    homogeneity = fast_glcm_homogeneity(glcm_input)
    contrast = fast_glcm_contrast(glcm_input)
    glcm_mean = fast_glcm_mean(glcm_input)
    entropy = fast_glcm_entropy(glcm_input)
    dissimilarity = fast_glcm_dissimilarity(glcm_input)

    # test with RGB normalization
    # b02 = (((b02 - b02.min()) / (b02.max() - b02.min()) )*255).astype('uint8')
    # b03 = (((b03 - b03.min()) / (b03.max() - b03.min()) )*255).astype('uint8')
    # b04 = (((b04 - b04.min()) / (b04.max() - b04.min()) )*255).astype('uint8')

    b01 = ((b01 - b01.min()) / (b01.max() - b01.min()) )
    b02 = ((b02 - b02.min()) / (b02.max() - b02.min()) )
    b03 = ((b03 - b03.min()) / (b03.max() - b03.min()) )
    b04 = ((b04 - b04.min()) / (b04.max() - b04.min()) )
    b05 = ((b05 - b05.min()) / (b05.max() - b05.min()) )
    b06 = ((b06 - b06.min()) / (b06.max() - b06.min()) )
    b07 = ((b07 - b07.min()) / (b07.max() - b07.min()) )
    b08 = ((b08 - b08.min()) / (b08.max() - b08.min()) )
    b8A = ((b8A - b8A.min()) / (b8A.max() - b8A.min()) )
    b09 = ((b09 - b09.min()) / (b09.max() - b09.min()) )
    b11 = ((b11 - b11.min()) / (b11.max() - b11.min()) )
    b12 = ((b12 - b12.min()) / (b12.max() - b12.min()) )

    #allbands = (allbands - allbands.min()) / (allbands.max() - allbands.min())

    # stack all features together
    #allbands = np.dstack([b01, b02, b03, b04, b05, b11, b12, ndvi, ndwi, entropy])
    #allbands = np.dstack([b01, b02, b03, b04, b05, b06, b07, b08, b8A, b09, b11, b12, ndvi, ndwi, entropy])
    #allbands = np.dstack([b01, b02, b03, b04, b05, b06, b07, b08, b8A, b09, b11, b12, ndvi, ndwi, ndmi, ndbi, ndre1, homogeneity, contrast, entropy])
    #allbands = np.dstack([allbands, ndvi, ndwi, ndmi, ndbi, ndre1, homogeneity, contrast, entropy])

    #allbands = np.dstack([ndvi, ndwi, ndmi, ndbi, ndre1, homogeneity, contrast, entropy])
    
    # best model between years
    #allbands = np.dstack([ndvi, ndwi, ndmi, homogeneity, entropy])

    # seems also good!
    #allbands = np.dstack([ndvi, ndwi, homogeneity, entropy])
    allbands = np.dstack([ndvi, ndwi, ndre1, homogeneity, entropy])

    # GLCM features are important - without them the model is not performing well!
    #allbands = np.dstack([ndvi, ndwi, ndre1])

    # write out training data
    profile.update({
        "dtype": "float32",
        "count": allbands.shape[2],
        "nodata": -99999
    })

    allbands_img = reshape_as_raster(allbands)

    # for testing only - write band to file system
    with rio.open(f"../../../data/training/{composite_date}.tif", "w", **profile) as dest:
        dest.write(allbands_img)

    print(allbands.shape)

    # take care of nans
    # for reflectances NaNs -> 0
    allbands = np.nan_to_num(allbands, nan=-99999)
    
    # reclass = reclass_array(ndbi_array=ndbi, ndvi_array=ndvi)

    #cluster = cluster_arrays(ndvi, ndbi, ndwi, b01, b02, b03, b04, b05, b06, b07, b08, b8A, b11)
    #cluster = cluster_arrays(b01, b02, b03, b04, b05, b06, b07, b08, b8A, b11)


    if training == True:

        model_path = train(allbands, esa_landcover=esa_landcover)

        # add training dataset name to model meta
        model_meta_path = f"{os.path.splitext(model_path)[0]}.txt"
        
        with open(model_meta_path, 'a') as fp:
            fp.write(f"\n\ntraining dataset(s): {input_dataset}")

        print(model_path)

    elif training == False:
        
        classified, classified_probas = predict(
            allbands, 
            model_path=model_path
        )

        profile.update({
            "dtype": np.uint8,
            "count": classified.shape[0],
            "nodata": 255
        })

        #
        # stats
        #
        print(f"\nStats classified {composite_date}:\n")
        print(f"Vegetation & Cropland area ha: {(np.nansum((classified == 1))*100/10000) + (np.nansum((classified == 2))*100/10000)}")
        # print(f"Cropland area ha: {np.nansum((classified == 2))*100/10000}")
        print(f"Built up area ha: {np.nansum((classified == 3))*100/10000}")
        print(f"Other area ha: {np.nansum((classified == 0))*100/10000}")
        print(f"Total area ha: {classified.size*100/10000}")

        # esa landcover
        print("\nStats landcover 2020:\n")
        print(f"Vegetation & Cropland area ha: {(np.nansum((esa_landcover <= 40))*100/10000)}")
        print(f"Built up area ha: {(np.nansum((esa_landcover == 50))+np.nansum((esa_landcover == 60)))*100/10000}")
        print(f"Other area ha: {np.nansum((esa_landcover >= 70))*100/10000}")
        print(f"Total area ha: {esa_landcover.size*100/10000}")

        # # city cluster test
        # cluster = cluster_arrays(classified.astype("uint8"))
        # print(cluster.shape)

        # for testing only - write band to file system
        with rio.open(f"../../../data/classified/{composite_date}.tif", "w", **profile) as dest:
            dest.write(classified)

        profile.update({
            "dtype": np.float32,
            "count": 1,
            "nodata": -1
        })
 
        for cl in classified_probas.keys():
            class_array = classified_probas.get(cl)
            
            with rio.open(f"../../../data/classified/{composite_date}_{cl}_proba.tif", "w", **profile) as dest:
                dest.write(class_array, 1)