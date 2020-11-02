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
import argparse, yaml
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from rasterio.features import shapes
import rasterio
from descartes.patch import PolygonPatch
from rasterio import Affine
import fiona
from fiona.crs import from_epsg

from sentinel_imagery_generator import SentinelImageryGenerator
from prediction_layer_generator import FeatureLayerGenerator
from utils import dilate, erode, vectorize

fiona.supported_drivers['KML'] = 'rw'
 

def get_args():
    parser = argparse.ArgumentParser(
        description= 'Predict irrigation presence using Sentinel imagery')

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
    '''
    Load pretrained model and normaliziation array. Compile and return.
    '''
    
    model = load_model(f'pretrained_models/models/{args.model_filename}')
    model_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)

    model.compile(model_optimizer)
    
    norm_array = pd.read_csv(f'pretrained_models/normalization_arrays/{args.norm_filename}.csv', index_col=0)
    norm_means = [float(i) for i in norm_array[f'{args.pred_region}_standard_array'].iloc[0].strip('[] ').split(',')]
    norm_stds  = [float(i) for i in norm_array[f'{args.pred_region}_standard_array'].iloc[1].strip('[] ').split(',')]
    
    return model, norm_means, norm_stds



def predict_for_tile(args, model, norm_means, norm_stds, feature_stack, valid_pixels, dltile):
    '''
    Predict irrigation presence over a specific DL Tile.
    '''    
    
    # Create file save string and folder
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
    
    # Create tensorflow dataset holding feature layers for predictions
    feature_stack_ds = tf.data.Dataset.from_tensor_slices(feature_stack_valid).batch(256)
    predictions_list = []
    
    for features in feature_stack_ds:
        predictions = model(features, training=False)
        predictions = tf.squeeze(predictions, axis=1)
        predictions = tf.argmax(predictions, axis=1)
        predictions_list.extend(predictions.numpy())

    # Reconstitute prediction map    
    prediction_layer[valid_pixel_locs] = np.array(predictions_list)
    
    # Morphological processing + vectorizing
    dilated = dilate(prediction_layer, iters=1)
    eroded = erode(dilated, iters=1)
    out_array, polys, polys_geo = vectorize(eroded, dltile)
    
    # Create GDF with polygons
    geoms_gdf = gpd.GeoDataFrame.from_features(polys_geo, crs=dltile.crs)
    geoms_gdf.crs = dltile.crs
    
    # Save GDF if polygons exist
    outfile = f'{pred_folder}/{args.model_filename}_eroded_preds_{dltile_key}_polys.kml'
    if len(geoms_gdf) > 0:
        geoms_gdf.to_file(outfile, driver="KML")
    
    ## Plot predictions on top of RGB Sentinel image
    fig, ax = plt.subplots()
    ax.imshow(rgb_image)
    
    # Add scale bar
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
    
    # Iterate through polygons, plot
    for ix, g in enumerate(polys):
        xy_tuples = g['geometry']['coordinates'][0]
        x,y = zip(*xy_tuples)
        ax.plot(x,y, 'r', linewidth=0.8)

    ## Save plots
    plt.savefig(f'{pred_folder}/{args.model_filename}_preds_{dltile_key}.png',  
                bbox_inches='tight', pad_inches=0)
    plt.imsave(f'{pred_folder}/{args.model_filename}_eroded_preds_{dltile_key}.png', eroded)
    plt.imsave(f'{pred_folder}/{args.model_filename}_dilated_preds_{dltile_key}.png', dilated)
    plt.imsave(f'{pred_folder}/{args.model_filename}_rgb_{dltile_key}.png', rgb_image)
    plt.imsave(f'{pred_folder}/{args.model_filename}_valid_{dltile_key}.png', valid_pixels[0])
      

    
        
if __name__ == '__main__':
    
    # Get args
    args = get_args()
    
    # Load a pretrained model + normalization
    model, norm_means, norm_stds = load_model_and_norm(args)
    
    # Specify the DL Tile if you have the particular key
#     dltile_key = '256:0:10.0:37:39:437'
#     dltile = dl.scenes.geocontext.DLTile.from_key(dltile_key)
#     print(dltile.geometry.centroid)
  
        
    # Load a DL Tile from a lat and lon
    lat = 12.8769
    lon = 37.7894
    dltile = dl.scenes.geocontext.DLTile.from_latlon(lat, lon, resolution = 10, 
                                                     tilesize = 256, pad = 0)
    
    # Create a sentinel timeseries generator object
    generator = SentinelImageryGenerator(args, dltile)
    
    
    # Retrieve Sentinel imagery, interpolate and smooth
    sentinel_file_name = f'data/raw_data/imagery_stacks/sentinel_{dltile.key}.npy'
    
    # Collect Sentinel timeseries stack if you haven't already saved it.
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
    
    # Load previously saved Sentinel stack 
    sentinel_stack = np.load(sentinel_file_name)

    # Create feature layer generator
    feature_generator = FeatureLayerGenerator(args, 
                                              sentinel_stack,
                                              chirps_array, 
                                              srtm_layer,
                                              gfsad_layer)
    
    # Define feature stack names
    feature_stack_file_name = f'data/processed_data/feature_layers/features_{dltile.key}.npy'
    ratio_file_name = f'data/processed_data/evi_max_min_ratio_layers/ratio_{dltile.key}.npy'    
    force_feature_stack_dl = True
    
    # If these files don't exist, create them
    if not os.path.exists(feature_stack_file_name) or force_feature_stack_dl:
        print('Creating new feature layers')
        feature_stack, evi_max_min_ratio = feature_generator.create_features()
        np.save(feature_stack_file_name, feature_stack)
        np.save(ratio_file_name, evi_max_min_ratio)
    
    # Load presaved files
    feature_stack = np.load(feature_stack_file_name)
    evi_max_min_ratio = np.load(ratio_file_name)
    
    # Find valid pixels for prediction
    valid_pixels = feature_generator.return_valid_pixels(evi_max_min_ratio)
    
    # Predict over the DL tile
    predict_for_tile(args, model, norm_means, norm_stds, feature_stack, valid_pixels, dltile)
   
    