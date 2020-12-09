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
from main import get_args, load_model_and_norm, predict_for_tile
from raw_imagery_generator import RawImageryGenerator
from descarteslabs.catalog import Product, Image, OverviewResampler
from feature_layer_generator import FeatureLayerGenerator
from utils import (save_model_to_dlstorage, save_normalization_to_dlstorage, 
                   load_model_from_dlstorage, load_normalization_from_dlstorage,
                  compile_model_clean_standardization, dotdict)




def load_polygons():
    folder_dir = '../data/raw_data/shapefiles/shapefiles_for_model_testing'
    irrig_polys = ['dabat_irrig_polygons.geojson',
                  'fincha_irrig_polygons.geojson',
                  'kobo_irrig_polygons.geojson',
                  'rift_irrig_polygons.geojson']
    
    noirrig_polys = ['dabat_noirrig_polygons.geojson',
                  'fincha_noirrig_polygons.geojson',
                  'kobo_noirrig_polygons.geojson',
                  'rift_noirrig_polygons.geojson']
    

    irrig_poly_list = []
    noirrig_poly_list = []
    
    for file in irrig_polys:
        loaded_poly = gpd.read_file(f'{folder_dir}/{file}').to_crs('EPSG:4326')
        irrig_poly_list.extend(loaded_poly['geometry'])
    
    for file in noirrig_polys:
        loaded_poly = gpd.read_file(f'{folder_dir}/{file}').to_crs('EPSG:4326')
        noirrig_poly_list.extend(loaded_poly['geometry'])
    
    
    return irrig_poly_list, noirrig_poly_list

def load_single_polygon():
    poly_path = '../data/raw_data/shapefiles/shapefiles_for_model_testing/gondar_large_polygon.geojson'
    poly = gpd.read_file(poly_path).to_crs('EPSG:4326')
    
    return poly['geometry']
    

def find_total_area(irrig_poly_list, noirrig_poly_list):
    
    irrig_area = 0 # sq km
    noirrig_area = 0 # sq km
    
    for poly in irrig_poly_list:
        irrig_area += poly.area*(111**2)
    
    for poly in noirrig_poly_list:
        noirrig_area += poly.area*(111**2)
    
    
    print(f'Irrig area: {irrig_area}')
    print(f'No-irrig area: {noirrig_area}')
    
    
    

def find_aois_by_tile(poly_list):
    print('Finding AOIs by tile')
    
    dltiles_list = []
    
    for poly in tqdm(poly_list):
        dltiles = DLTile.from_shape(poly, resolution=10, tilesize=256, pad=0)
        dltiles_list.extend(dltiles)
        
    return dltiles_list



def deploy_feature_creation(args, dltile_key):
    
    import descarteslabs as dl
    import os
    from descarteslabs.catalog import Product, Image, OverviewResampler
    from scripts.raw_imagery_generator import RawImageryGenerator
    from scripts.feature_layer_generator import FeatureLayerGenerator

    dltile = dl.scenes.DLTile.from_key(dltile_key)    
    
    generator = RawImageryGenerator(args, dltile)

    # Retrieve Sentinel stack
    sentinel_stack = generator.find_sentinel_imagery()
    sentinel_stack = generator.temporal_interp_and_smoothing(sentinel_stack)

    # Retrieve CHIRPS predictions
    chirps_array = generator.find_chirps_imagery()

    # Retrieve SRTM layer
    srtm_layer = generator.return_srtm_layer()

    # Retrieve GFSAD LULC layer
    gfsad_layer = generator.find_lulc_imagery()

    feature_generator = FeatureLayerGenerator(args, 
                                          sentinel_stack,
                                          chirps_array, 
                                          srtm_layer,
                                          gfsad_layer)

    feature_stack, evi_max_min_ratio = feature_generator.create_features()

    upload = True
    if upload:
        pid = 'dry_crop_feats'
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
            feature_stack,
            overviews=[2, 4],
            overview_resampler=OverviewResampler.MODE,
            overwrite=True
        )

        upload.wait_for_completion()

    return


def deploy_on_tasks():
    docker_image = 'us.gcr.io/dl-ci-cd/images/tasks/public/py3.8:v2020.09.22-5-ga6b4e5fa'
    
    tasks = dl.Tasks()
    async_predict = tasks.create_function(
        f=deploy_feature_creation,
        name='veg_prediction-deploy',
        image=docker_image,
        maximum_concurrency=500,
        memory="3Gi",
        include_modules = ['scripts'],
#         requirements = ['scipy==1.4.1',  'pyfftw==0.12.0']
    )
    
    return async_predict
                

if __name__ == '__main__':
    
    
    args = get_args()
    args = dotdict(vars(args))
    
    gondar_poly = load_single_polygon()

    dltiles_list = find_aois_by_tile([gondar_poly])
    
    print(len(dltiles_list))

    docker_image = 'us.gcr.io/dl-ci-cd/images/tasks/public/py3.8:v2020.09.22-5-ga6b4e5fa'
    
    tasks = dl.Tasks()
    async_predict = deploy_on_tasks()

    
    for tile in dltiles_list[0:5]:
        print(tile.key)
        async_predict(args, tile.key)
    
    

    