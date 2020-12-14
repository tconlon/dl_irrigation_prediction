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


def load_polygons():
    folder_dir = '../data/raw_data/shapefiles/shapefiles_for_model_testing'
    irrig_polys = ['dabat_irrig_polygons.geojson',
                  ]
    
    noirrig_polys = ['dabat_noirrig_polygons.geojson',
                  ]
    

    irrig_poly_list = []
    noirrig_poly_list = []
    
    for file in irrig_polys:
        loaded_poly = gpd.read_file(f'{folder_dir}/{file}').to_crs('EPSG:4326')
        irrig_poly_list.extend(loaded_poly['geometry'])
    
    for file in noirrig_polys:
        loaded_poly = gpd.read_file(f'{folder_dir}/{file}').to_crs('EPSG:4326')
        noirrig_poly_list.extend(loaded_poly['geometry'])
    
    
    return irrig_poly_list, noirrig_poly_list


def calculate_accuracy(poly_list, positive_class=True):
    
    acc_list = []
    total_pixels_list = []
    
    for poly in tqdm(poly_list):
        aoi = dl.scenes.geocontext.AOI(poly, crs='EPSG:32637', resolution=10)
        
        pred_scenes, ctx =  dl.scenes.search(
            aoi,
            products='columbia_sel:dry_season_crop_model_preds',
            limit=None,
        )
                
        if len(pred_scenes) > 0:    
            pred_mosaic = pred_scenes.mosaic(bands='veg_growth_pred_probability',
                                        ctx=aoi)
            
            rounded_preds = np.around(pred_mosaic)
            pixel_ct = rounded_preds.count()
            acc = np.count_nonzero(rounded_preds==int(positive_class)) / pixel_ct
            
            acc_list.append(acc)
            total_pixels_list.append(pixel_ct)
            
        else:
            print('No overlap')  
    
    weighted_acc = np.sum(np.array(acc_list) * np.array(total_pixels_list))/np.sum(total_pixels_list)
    
    print(f'Weighted class accuracy: {weighted_acc}')
    
            
if __name__ == '__main__':
            
    irrig_poly_list, noirrig_poly_list = load_polygons()
    
    calculate_accuracy(irrig_poly_list, positive_class=True)
    calculate_accuracy(noirrig_poly_list, positive_class=False)
    
    