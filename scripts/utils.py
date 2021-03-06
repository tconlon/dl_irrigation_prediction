import numpy as np
from rasterio import Affine
import rasterio
from scipy.ndimage import convolve

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def dilate(map_layer, iters):
    '''
    Dilate positive predictions by defined kernel.
    '''
    
    kernel = np.array([[0., 1., 0.],
                        [ 1., 1., 1.],
                        [ 0., 1., 0.]])

    for i in range(iters):
        map_layer = map_layer * 1.0
        map_layer = convolve(map_layer, kernel) > 0

    return map_layer

def erode(map_layer, iters):
    '''
    Erode positive predictions by defined kernel.
    '''
    
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
    '''
    Vectorize positive predictions.
    '''
    
    img = img.astype(np.int16)
    
    # Remove clusters of predictions smaller than ~size~ pixels.
    out_array = rasterio.features.sieve(img, size=16)
    
    # Define an affine transformation
    affine_trans = Affine(dltile.geotrans[1], dltile.geotrans[2], dltile.geotrans[0],
                         dltile.geotrans[4], dltile.geotrans[5], dltile.geotrans[3])
    
    # Generate polygons from the cleaned output array
    polygons = rasterio.features.shapes(out_array) 
    
    # Generate polygons from the cleaned ouput array with a transform defined.
    polygons_geo = rasterio.features.shapes(out_array, transform=affine_trans)
    
    # Organize in lists of dictionaries.
    polys = [
        {"properties": {"raster_val": v}, "geometry": s, "type": "feature"}
        for s, v in polygons if v == 1
    ]
    
    polys_geo = [
        {"properties": {"raster_val": v}, "geometry": s, "type": "feature"}
        for s, v in polygons_geo if v == 1
    ]
    
    return out_array, polys, polys_geo


def clipping_to_aoi():
    
    
    for dltile in dltiles_for_poly:
        ctx = dltile.geometry
        intersection = ctx.intersection(MultiPolygon(poly_list))
        aoi_intersect = AOI(intersection, resolution=10, crs=dltile.crs, shape=(64, 64))

        aoi_intersect_list.append(aoi_intersect)

        

def save_model_to_dlstorage():
    '''
    Save trained generator to DL Storage under 'model_name'
    Specify the generator for upload by assigning 'folder_to_save' to the folder
    that contains the saved generator.
    '''
    import os
    import shutil
    import descarteslabs as dl
    
    model_name = 'best_trained_all_regions'
    
    dir_name = f'../pretrained_models/models/{model_name}'
    output_filename = model_name
    shutil.make_archive(model_name, 'zip', dir_name)
    
   
    print(f'Upload model to DL Storage: {model_name}')
    storage = dl.Storage()
    storage.set_file(model_name, f'{model_name}.zip')
        
    os.remove(f'{model_name}.zip')    
    

def save_normalization_to_dlstorage():
    '''
    Save normalization .csv containing band means + standard deviations
    to DL Storage
    '''
    import descarteslabs as dl
    import os
    
    file_name = 'all_regions_norm_array'
    norm_file_path = f'../pretrained_models/normalization_arrays/{file_name}.csv'
    
    print(f'Upload normalization to DL Storage: {file_name}')
    storage = dl.Storage()
    storage.set_file(file_name, norm_file_path)

   
        
def load_model_from_dlstorage():
    '''
    Load TF model from DL.Storage
    '''
    import tensorflow as tf
    import tempfile
    from tensorflow.keras.models import load_model
    import descarteslabs as dl
    import os
    import subprocess
    
    storage_key = 'best_trained_all_regions'

    try: 
        model_zip = tempfile.NamedTemporaryFile()
        model_dir = tempfile.TemporaryDirectory()
        
        dl.Storage().get_file(storage_key, model_zip.name)
         
        unzip = f'unzip {model_zip.name} -d {model_dir.name}'
        resp = os.system(unzip)

        model = load_model(model_dir.name)
        model_zip.close()
        model_dir.cleanup()

        return model
    
    except:
        print("Model not available in DL Storage")

    

def load_normalization_from_dlstorage():
    '''
    Load the normalization mean and standard deviation from storage
    '''
    import descarteslabs as dl
    import pandas as pd
    import os
    
    storage_key = 'all_regions_norm_array'
    
    try:
        std_file = 'band_standardization.csv'

        dl.Storage().get_file(storage_key, std_file)

        std_metrics = pd.read_csv(std_file, index_col=0)
        os.remove(std_file)

        return std_metrics
    
    except:
        print("Standardizations not available in DL Storage")

def compile_model_clean_standardization(args, model, std_metrics):
    import tensorflow as tf
    
    
    model_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
    model.compile(model_optimizer)
    
    norm_means = [float(i) for i in std_metrics[f'{args.pred_region}_standard_array'].iloc[0].strip('[] ').split(',')]
    norm_stds  = [float(i) for i in std_metrics[f'{args.pred_region}_standard_array'].iloc[1].strip('[] ').split(',')]
    
    return model, np.array(norm_means), np.array(norm_stds)

def find_feature_layers_and_standardize(dltile_key, norm_means, norm_stds):
    import descarteslabs as dl
    import numpy as np
    
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
    import descarteslabs as dl
    import tensorflow as tf
    import numpy as np

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
