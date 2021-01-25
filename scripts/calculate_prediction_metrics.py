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


def load_polygons(aois_list):
    
    full_irrig_poly_list = []
    full_noirrig_poly_list = []
    
    for aoi in aois_list:
    
        folder_dir = '../data/raw_data/shapefiles/shapefiles_for_model_testing'
        irrig_polys = f'{aoi}_irrig_polygons.geojson',
                  
    
        noirrig_polys = f'{aoi}_noirrig_polygons.geojson',
                  
    

        aoi_irrig_poly_list = []
        aoi_noirrig_poly_list = []
    
        for file in irrig_polys:
            loaded_poly = gpd.read_file(f'{folder_dir}/{file}').to_crs('EPSG:4326')
            aoi_irrig_poly_list.extend(loaded_poly['geometry'])

        for file in noirrig_polys:
            loaded_poly = gpd.read_file(f'{folder_dir}/{file}').to_crs('EPSG:4326')
            aoi_noirrig_poly_list.extend(loaded_poly['geometry'])
    
        full_irrig_poly_list.append(aoi_irrig_poly_list)
        full_noirrig_poly_list.append(aoi_noirrig_poly_list)
    
    
    return full_irrig_poly_list, full_noirrig_poly_list


def calculate_accuracy(aois):
    
    irrig_poly_list, noirrig_poly_list = load_polygons(aois)

    results_df = pd.DataFrame()
    
    preds_name = 'dspc_preds_model_20201217-141812_allregions'
    row_labels = ['Total Pixels', 'Base case (No filtering)', 'SRTM + EVI 1.5 filtering',
                  'SRTM + EVI 2 filtering', 'SRTM, EVI 1.5, + AfSIS cropland filtering',
                  'SRTM, EVI 1.5, + ESA cropland filtering', 
                  'SRTM, EVI 1.5, + Copernicus cropland filtering',
                  'SRTM, EVI 1.5, + GFSAD cropland filtering']
    
    results_df['scenario'] = row_labels

  
    for ix, aoi_name in enumerate(aois):
        print(aoi_name)
        
        aoi_polys = [irrig_poly_list[ix], noirrig_poly_list[ix]]
    
        for jx, poly_list in enumerate(aoi_polys):
            print(jx)
            
            if jx == 0:
                col = f'{aoi_name}_dpsc_pos'
                target = 1
            else:
                col = f'{aoi_name}_dpsc_neg'
                target = 0
                
            correct_list = []
            total_pixels_list = []
        

            for poly in tqdm(poly_list):
                aoi = dl.scenes.geocontext.AOI(poly, crs='EPSG:32637', resolution=10)

                pred_scenes, ctx = dl.scenes.search(
                    aoi,
                    products=f'columbia_sel:{preds_name}',
                    limit=None,
                )

                feat_scenes, ctx = dl.scenes.search(
                    aoi,
                    products='columbia_sel:dry_crop_feats',
                    limit=None,
                )

                markus_scenes, ctx = dl.scenes.search(
                    aoi,
                    products='columbia_sel:afsis_cropland_predictions',
                    limit=None,
                )

                esa_scenes, ctx = dl.scenes.search(
                    aoi,
                    products='columbia_sel:esa_cci_lulc_predictions',
                    limit=None,
                )

                copernicus_scenes, ctx = dl.scenes.search(
                    aoi,
                    products='columbia_sel:copernicus_lulc_predictions',
                    limit=None,

                )

                gfsad_scenes, ctx = dl.scenes.search(
                    aoi,
                    products='usgs:gfsad30:global:v1',
                    limit=None,

                )


                if len(pred_scenes) > 0:    
                    pred_mosaic = pred_scenes.mosaic(bands='dspc_pred_probability',
                                                ctx=aoi)


                    feat_mosaic = feat_scenes.mosaic(bands='srtm evi_max_min_ratio_90_10',
                                                    ctx=aoi)

                    markus_mosaic = markus_scenes.mosaic(bands='pred_cp_mask',
                                                        ctx=aoi)

                    esa_mosaic = esa_scenes.mosaic(bands='lulc_prediction',
                                                  ctx=aoi)

                    copernicus_mosaic = copernicus_scenes.mosaic(bands='lulc_prediction',
                                                                ctx=aoi)

                    gfsad_mosaic = gfsad_scenes.mosaic(bands='Land_Cover', ctx=aoi)

                    valid_pixels = (~pred_mosaic.mask* ~markus_mosaic.mask * ~esa_mosaic.mask *
                                    ~copernicus_mosaic.mask * ~gfsad_mosaic.mask)

                    mx_valid = np.ma.masked_array(valid_pixels.astype(int), mask=~valid_pixels)

                    feat_valid = (feat_mosaic[0] < 800) * (feat_mosaic[1] > 1.5)
                    feat_valid_2 = (feat_mosaic[0] < 800) * (feat_mosaic[1] > 2)
                    esa_valid  = np.isin(esa_mosaic, [4,5])
                    copernicus_valid = np.isin(copernicus_mosaic, [40, 90])
                    markus_valid = np.isin(markus_mosaic, [1])
                    gfsad_valid = np.isin(gfsad_mosaic, [2])


                    rounded_preds = np.around(pred_mosaic)
                    pixel_ct = mx_valid.count()

                    base_correct = np.count_nonzero(mx_valid*rounded_preds==target)

                    feat_correct = np.count_nonzero((mx_valid*rounded_preds*feat_valid) == 
                                                    target)
                    
                    feat_correct_2 = np.count_nonzero((mx_valid*rounded_preds*feat_valid_2) == 
                                                    target)

                    feat_markus_correct = np.count_nonzero((mx_valid*rounded_preds*feat_valid*markus_valid) == 
                                                       target)

                    feat_esa_correct = np.count_nonzero((mx_valid*rounded_preds*feat_valid*esa_valid) == 
                                                       target)

                    feat_copernicus_correct = np.count_nonzero((mx_valid*rounded_preds*feat_valid*copernicus_valid) == 
                                                       target)

                    feat_gfsad_correct = np.count_nonzero((mx_valid*rounded_preds*feat_valid*gfsad_valid) == 
                                                       target)


                    correct_list.append([base_correct, feat_correct, feat_correct_2, 
                                         feat_markus_correct, feat_esa_correct, 
                                         feat_copernicus_correct, feat_gfsad_correct])
                    total_pixels_list.append(pixel_ct)

                else:
                    print('No overlap')
                    
#                     correct_list.append([0, 0, 0, 0, 0, 0, 0])
#                     total_pixels_list.append(0)
                    


            scen_accs = np.zeros(len(correct_list[0])+1)
            tot_pix = np.sum(total_pixels_list)
            
            scen_accs[0] = tot_pix
            
            for ix in range(len(correct_list[0])):
                scen_accs[ix+1] = np.sum([correct_list[jx][ix] for jx in 
                                        range(len(correct_list))])/tot_pix

            results_df[col] = scen_accs
            
    results_df.to_csv(f'../results/sentinel_prediction_evaluation_{preds_name}.csv')
            
           
            
if __name__ == '__main__':
            
    aois = ['gondar', 'rift', 'fincha', 'kobo']
        
    calculate_accuracy(aois)
    