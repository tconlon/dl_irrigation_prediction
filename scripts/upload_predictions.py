import tensorflow as tf
import numpy as np
import sys, os
from tqdm import tqdm
from glob import glob
import pandas as pd
import geopandas as gpd
import descarteslabs as dl
from descarteslabs.scenes import SceneCollection
from descarteslabs.scenes.geocontext import AOI, DLTile
import shapely
from shapely.geometry import MultiPolygon 
from scripts.main import get_args, load_model_and_norm, predict_for_tile
from scripts.raw_imagery_generator import RawImageryGenerator
from descarteslabs.catalog import Product, Image, OverviewResampler
from upload_feature_layers import load_single_polygon, find_aois_by_tile
from feature_layer_generator import FeatureLayerGenerator
from utils import (save_model_to_dlstorage, save_normalization_to_dlstorage, 
                   load_model_from_dlstorage, load_normalization_from_dlstorage,
                  compile_model_clean_standardization, dotdict)



def find_feature_layers_and_standardize(dltile_key, norm_means, norm_stds):
    print('Find feature layers')
    dltile = dl.scenes.DLTile.from_key(dltile_key)
    
    dl_scenes, ctx = dl.scenes.search(
            dltile,
            products='columbia_sel:dry_crop_feats',
            start_datetime='2019-12-31',
            end_datetime='2020-01-02',
            limit=None,
    )
    
    bands_to_extract = ['srtm', 'evi_annual_corrcoef', 'evi_chirps_corrcoef',
                        'evi_at_min_12_chirps_mean', 'evi_at_min_24_chirps_mean', 
                        'evi_at_min_36_chirps_mean', 'evi_at_min_12_chirps_max', 
                        'evi_at_min_24_chirps_max', 'evi_at_min_36_chirps_max',
                        'evi_max_min_ratio_95_5', 'evi_max_min_ratio_90_10', 
                        'evi_max_min_ratio_85_15', 'evi_max_min_ratio_80_20',
                        'ndwi_annual_corrcoef', 'ndwi_chirps_corrcoef',
                        'ndwi_at_min_12_chirps_mean', 'ndwi_at_min_24_chirps_mean', 
                        'ndwi_at_min_36_chirps_mean', 'ndwi_at_min_12_chirps_max', 
                        'ndwi_at_min_24_chirps_max', 'ndwi_at_min_36_chirps_max',]
    
    
    feature_layers = dl_scenes.mosaic(bands=bands_to_extract,
                                      ctx=ctx)
        
    feature_layers = np.transpose(feature_layers, (1,2,0))
    feature_layers = np.reshape(feature_layers, (feature_layers.shape[0]*feature_layers.shape[1],
                                                feature_layers.shape[2]))
    
    feature_layers = (feature_layers - norm_means)/norm_stds
    
    
    return feature_layers
    

def predict_function(model, norm_means, norm_stds, dltile_key):

    dltile = dl.scenes.DLTile.from_key(dltile_key)
      
    feature_layers = find_feature_layers_and_standardize(dltile_key, norm_means, norm_stds)
    feature_layers = np.expand_dims(feature_layers, axis = 1)
    
    feature_stack_ds = tf.data.Dataset.from_tensor_slices(feature_layers).batch(256)
    predictions_list = []
    
    print('Predict over DLTile')
    for features in feature_stack_ds:
        predictions = model(features, training=False)
        predictions = tf.squeeze(predictions, axis=1)
        predictions = predictions[:,1]
        #predictions = tf.argmax(predictions, axis=1)
        predictions_list.extend(predictions.numpy())
    
    
    predictions_over_tile = np.reshape(np.array(predictions_list), (1, dltile.tilesize,
                                                                   dltile.tilesize))
    
    return predictions_over_tile
    
def deploy_predictions(args, dltile_key):
    
    import descarteslabs as dl
    import os
    from descarteslabs.catalog import Product, Image, OverviewResampler
    from utils import (load_model_from_dlstorage, load_normalization_from_dlstorage,
                      compile_model_clean_standardization, predict_function,
                      find_feature_layers_and_standardize, dotdict)

#     import predict_function, find_feature_layers_and_standardize
    
    args = dotdict(args)
    
    ## Load in model and normalizations
    model = load_model_from_dlstorage()
    std_metrics = load_normalization_from_dlstorage()
    model, norm_means, norm_stds = compile_model_clean_standardization(args, model, std_metrics)
    dltile = dl.scenes.DLTile.from_key(dltile_key)    
    
    predictions_over_tile = predict_function(model, norm_means, norm_stds, dltile_key)
    
    
    upload = True
    if upload:
        print('Upload to catalog')
        pid = 'dry_season_crop_model_preds'
        auth = dl.Auth()
        pid = f"{auth.payload['org']}:{pid}"
        product = Product.get(pid)

        img_name = dltile.key.replace(':', '_')

        img_id = f"{pid}:{img_name}"
        image = Image(product=product, name=img_name, id=img_id)
        image.acquired = '2020-01-01'
        image.geotrans = dltile.geotrans
        image.cs_code = dltile.crs

        upload = image.upload_ndarray(
            predictions_over_tile,
            overviews=[2, 4],
            overview_resampler=OverviewResampler.MODE,
            overwrite=True
        )

        upload.wait_for_completion()

    return


def deploy_on_tasks(args):
    docker_image = 'us.gcr.io/dl-ci-cd/images/tasks/public/py3.8:v2020.09.22-5-ga6b4e5fa'
    
    tasks = dl.Tasks()
    async_predict = tasks.create_function(
        f=deploy_predictions,
        name='veg_prediction-deploy',
        image=docker_image,
        maximum_concurrency=500,
        memory="3Gi",
        include_modules = ['utils'],
        requirements = ['scipy==1.4.1',  'pyfftw==0.12.0']
    )
    
        
    return async_predict
                

        
        
if __name__ == '__main__':
    
    
    args = get_args()
    args = dotdict(vars(args))

    dltile_key = '256:0:10.0:37:-66:542'
    
    
    gondar_poly = load_single_polygon()
    dltiles_list = find_aois_by_tile([gondar_poly])
    
    print(len(dltiles_list))
    
    async_predict = deploy_on_tasks(args)
    
    for dltile in dltiles_list[4::]:
        async_predict(args, dltile.key)