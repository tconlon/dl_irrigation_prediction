import tensorflow as tf
import numpy as np
import sys, os
from tqdm import tqdm
from glob import glob
import pandas as pd
import geopandas as gpd

import descarteslabs as dl
from descarteslabs.scenes import SceneCollection
from appsci_utils.image_processing.coregistration import coregister_stack
import random
import datetime
from tqdm import tqdm
from sentinel_imagery_generator import SentinelImageryGenerator
from prediction_layer_generator import FeatureLayerGenerator
import argparse, yaml
from tensorflow.keras.models import load_model
from scipy.ndimage import convolve

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from rasterio.features import shapes
import rasterio
from descartes.patch import PolygonPatch
from rasterio import Affine
import fiona
from fiona.crs import from_epsg


fiona.supported_drivers['KML'] = 'rw'
print(gpd.__version__)


def get_args():
    parser = argparse.ArgumentParser(
        description= 'Predict irrigation presence using sentinel imagery')

    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help='Filename defining repo configuration')

    args = parser.parse_args()
    config = yaml.load(open(args.params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    return args

def load_model_and_norm(args):
    
    model = load_model(f'pretrained_models/models/{args.model_filename}')
    model_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)

    model.compile(model_optimizer)
    
    norm_array = pd.read_csv(f'pretrained_models/normalization_arrays/{args.norm_filename}.csv', index_col=0)
    norm_means = [float(i) for i in norm_array[f'{args.pred_region}_standard_array'].iloc[0].strip('[] ').split(',')]
    norm_stds  = [float(i) for i in norm_array[f'{args.pred_region}_standard_array'].iloc[1].strip('[] ').split(',')]
    
    return model, norm_means, norm_stds


def dilate(map_layer, iters):
    kernel = np.array([[0., 1., 0.],
                        [ 1., 1., 1.],
                        [ 0., 1., 0.]])

    for i in range(iters):
        map_layer = map_layer * 1.0
        map_layer = convolve(map_layer, kernel) > 0

    return map_layer

def erode(map_layer, iters):
    map_layer = ~map_layer

    kernel = np.array([[0., 1., 0.],
                        [ 1., 1., 1.],
                        [ 0., 1., 0.]])

    for i in range(iters):
        map_layer = map_layer * 1.0
        map_layer = convolve(map_layer, kernel) > 0

    map_layer = ~map_layer

    return map_layer

def vectorize(img, dltile):
    img = img.astype(np.int16)
    
    out_array = rasterio.features.sieve(img, size = 16)
    
    affine_trans = Affine(dltile.geotrans[1], dltile.geotrans[2], dltile.geotrans[0],
                         dltile.geotrans[4], dltile.geotrans[5], dltile.geotrans[3])
    
    
    polygons = rasterio.features.shapes(out_array) 
    polygons_geo = rasterio.features.shapes(out_array, transform=affine_trans) #, transform=transform)

    
    polys = [
        {"properties": {"raster_val": v}, "geometry": s, "type": "feature"}
        for s, v in polygons if v == 1
    ]
    
    polys_geo = [
        {"properties": {"raster_val": v}, "geometry": s, "type": "feature"}
        for s, v in polygons_geo if v == 1
    ]
    
    return out_array, polys, polys_geo
    


def predict_for_tile(args, model, norm_means, norm_stds, feature_stack, valid_pixels, dltile):
    # Transpose feature stack to retrieve valid pixel locations
    
    
    dltile_key = dltile.key.replace(':', '_')
    pred_folder = f'predictions/{dltile_key}'
    if not os.path.exists(pred_folder):
        os.mkdir(pred_folder)
    
    print('Create RGB image')
    
    rgb_image = generator.generate_true_color_image(pred_folder)
    
    print('Predict')
    
    feature_stack = np.transpose(feature_stack, (1,2,0))
    prediction_layer = np.zeros((feature_stack.shape[0], feature_stack.shape[1],)).astype(np.int16)
    
    valid_pixel_locs = np.where(valid_pixels[0])
    
    # Extract correct pixels
    feature_stack_valid = feature_stack[valid_pixel_locs]
    
    # Normalize and format
    feature_stack_valid = ((feature_stack_valid - norm_means)/norm_stds).astype(np.float32)
    feature_stack_valid = np.expand_dims(feature_stack_valid, axis = 1)
    

    feature_stack_ds = tf.data.Dataset.from_tensor_slices(feature_stack_valid).batch(256)
    predictions_list = []
    
    for features in feature_stack_ds:
        predictions = model(features, training=False)
        predictions = tf.squeeze(predictions, axis=1)
        predictions = tf.argmax(predictions, axis=1)
        predictions_list.extend(predictions.numpy())

        
    prediction_layer[valid_pixel_locs] = np.array(predictions_list)
    
    
    ## Morphological processing + vectorizing
    dilated = dilate(prediction_layer, iters=1)
    eroded = erode(dilated, iters=1)
    out_array, polys, polys_geo = vectorize(eroded, dltile)
    
    # Save outfile
    geoms_gdf = gpd.GeoDataFrame.from_features(polys_geo, crs=dltile.crs)
    geoms_gdf.crs = dltile.crs
    
    
    outfile = f'{pred_folder}/{args.model_filename}_eroded_preds_{dltile_key}_polys.kml'
    if len(geoms_gdf) > 0:
        geoms_gdf.to_file(outfile, driver="KML")
    
    ## PLOT
    fig, ax = plt.subplots()
    ax.imshow(rgb_image)
     
    fontprops = fm.FontProperties(size=12)
    bar_width = 50
    scalebar = AnchoredSizeBar(ax.transData,
                                           bar_width, '500m', 'lower right',
                                           pad=0.3,
                                           color='Black',
                                           frameon=True,
                                           size_vertical=2,
                                           fontproperties=fontprops)

    ax.add_artist(scalebar)
    
    for ix, g in enumerate(polys):
        
        xy_tuples = g['geometry']['coordinates'][0]
        x,y = zip(*xy_tuples)
        ax.plot(x,y, 'r', linewidth=0.8)

    
    plt.savefig(f'{pred_folder}/{args.model_filename}_preds_{dltile_key}.png',  bbox_inches='tight', pad_inches=0)
    
    ## Save plots
    plt.imsave(f'{pred_folder}/{args.model_filename}_eroded_preds_{dltile_key}.png', eroded)
    plt.imsave(f'{pred_folder}/{args.model_filename}_dilated_preds_{dltile_key}.png', dilated)
    plt.imsave(f'{pred_folder}/{args.model_filename}_rgb_{dltile_key}.png', rgb_image)
    plt.imsave(f'{pred_folder}/{args.model_filename}_valid_{dltile_key}.png', valid_pixels[0])
      

    
        
if __name__ == '__main__':
    
    args = get_args()
    
#     dltile_key = '256:0:10.0:37:39:437'
#     dltile = dl.scenes.geocontext.DLTile.from_key(dltile_key)
#     print(dltile.geometry.centroid)
  
    

    model, norm_means, norm_stds = load_model_and_norm(args)

    lat = 12.8769
    lon = 37.7894
    
    
    dltile = dl.scenes.geocontext.DLTile.from_latlon(lat, lon, resolution = 10, 
                                                     tilesize = 256, pad = 0)
    
    
    generator = SentinelImageryGenerator(args, dltile)
    
    
    # Retrieve Sentinel imagery, interpolate and smooth
    
    sentinel_file_name = f'data/raw_data/imagery_stacks/sentinel_{dltile.key}.npy'
    
    if not os.path.exists(sentinel_file_name):
        print('Finding new imagery')
        sentinel_stack = generator.find_sentinel_imagery()
        sentinel_stack = generator.temporal_interp_and_smoothing(sentinel_stack)
        np.save(sentinel_file_name, sentinel_stack)
    
    
    # Retrieve CHIRPS predictions
    chirps_array = generator.find_chirps_imagery()
    
    # Retrieve SRTM layer
    srtm_layer = generator.return_srtm_layer()

    # Retrieve GFSAD LULC layer
    gfsad_layer = generator.find_lulc_imagery()
    
#   ## Save for layer generation testing
        
    sentinel_stack = np.load(sentinel_file_name)

    # Create feature layer generator
    feature_generator = FeatureLayerGenerator(args, 
                                              sentinel_stack,
                                              chirps_array, 
                                              srtm_layer,
                                              gfsad_layer)
    
    # Generate stack of feature layers
    
    feature_stack_file_name = f'data/processed_data/feature_layers/features_{dltile.key}.npy'
    ratio_file_name = f'data/processed_data/evi_max_min_ratio_layers/ratio_{dltile.key}.npy'
    
    force_feature_stack_dl = True
    
    if not os.path.exists(feature_stack_file_name) or force_feature_stack_dl:
        print('Creating new feature layers')
        feature_stack, evi_max_min_ratio = feature_generator.create_features()
        np.save(feature_stack_file_name, feature_stack)
        np.save(ratio_file_name, evi_max_min_ratio)
    
    feature_stack = np.load(feature_stack_file_name)
    evi_max_min_ratio = np.load(ratio_file_name)
    
    # Find valid pixels for prediction
    valid_pixels = feature_generator.return_valid_pixels(evi_max_min_ratio)
    
    predict_for_tile(args, model, norm_means, norm_stds, feature_stack, valid_pixels, dltile)
   
    