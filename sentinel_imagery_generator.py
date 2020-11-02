import numpy as np
import logging; logging.getLogger().setLevel(logging.INFO); logging.captureWarnings(True)
import sys, os
from tqdm import tqdm
from glob import glob

import descarteslabs as dl
from descarteslabs.scenes import SceneCollection
from appsci_utils.regularization.spatiotemporal_denoise_stack import spatiotemporally_denoise
from appsci_utils.file_io.geotiff import write_geotiff
from appsci_utils.image_processing.coregistration import coregister_stack
import random
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from dateutil.rrule import rrule, MONTHLY
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


class SentinelImageryGenerator():

    def __init__(self, args, dltile):
        self.args = args
        
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.dltile = dltile
        self.max_cloud_frac = args.max_cloud_frac
        self.month_valid_start_day = args.month_valid_start_day
        self.month_valid_end_day = args.month_valid_end_day
        
        self.pad_for_interpolation = True
        self.smooth_temporally = True
        
    def find_sentinel_imagery(self):
        print(f'Loading imagery between {self.start_date} and {self.end_date}')
        
        s2_start_date = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        s2_end_date = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        
        
        s2_scenes, ctx = dl.scenes.search(
            self.dltile,
            products='sentinel-2:L1C',
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            cloud_fraction=self.max_cloud_frac,
            limit=None,
        )

        # Get Sentinel-2 cloud masks
#         s2_scenes_clouds, _ = dl.scenes.search(
#             self.dltile,
#             products='sentinel-2:L1C:dlcloud:v1',
#             start_datetime=self.start_date,
#             end_datetime=self.end_date,
#             limit=None,
#         )      
        
        s2_scenes = s2_scenes.filter(
                        lambda s: s.properties.date.day >= self.month_valid_start_day and
                                  s.properties.date.day <= self.month_valid_end_day
                    ).groupby(
                        'properties.date.year', 
                        'properties.date.month',
                    )
        
        
#         s2_scenes_clouds = s2_scenes_clouds.groupby('properties.date.year', 
#                                                     'properties.date.month', 
#                                                     )
        
        imagery_list = []
        
        dt_strt = datetime.datetime.strptime(self.args.start_date, '%Y-%m-%d')
        dt_end = datetime.datetime.strptime(self.args.end_date, '%Y-%m-%d')
        
        date_tuples = [(dt.year, dt.month) for dt in rrule(MONTHLY, 
                                                           dtstart=dt_strt, 
                                                           until=dt_end)]
        
        masked_array = np.ma.masked_all((len(date_tuples), self.dltile.tilesize, self.dltile.tilesize, 2))
            
#         print(f'Number of months available: {len(s2_scenes)}')    
            
        for  dt_tuple, month_scenes in s2_scenes:
            
            (year, month) = dt_tuple
            
            month_index = np.argwhere([i == dt_tuple for i in date_tuples])[0][0]
  
            
            sorted_scenes = month_scenes.sorted(lambda s: s.properties.cloud_fraction,
                                               reverse=True)
            
            # Only use the Sentinel cloud-mask, derived:visual_cloud_mask doesn't seem accutate
            stack = sorted_scenes.stack(bands = 'derived:evi derived:ndwi cloud-mask', 
                                          ctx = self.dltile,
                                         bands_axis=-1)
            
            # Extract derived layers + normalize
            stack_bands = (stack[..., 0:2] - 32767.5)/32767.5
            
            # Normalize
            stack_clouds = np.max(stack[..., 2:3], axis=-1, keepdims = True)
            
            stack_clouds_tiled = np.tile(stack_clouds, (1,1,1, stack_bands.shape[-1]))
            masked_stack = np.ma.array(stack_bands, mask = stack_clouds_tiled)
            masked_stack_median = np.ma.median(masked_stack, axis = 0)
                        
            masked_array[month_index] = masked_stack_median
                 
#         imagery_stack = np.ma.stack(imagery_list, axis = 0)

        print(masked_array.shape)
        return masked_array 


    def temporal_interp_and_smoothing(self, imagery_stack):
        print('Temporal interpolation and smoothing')
        if self.pad_for_interpolation:
            imagery_stack = np.ma.concatenate([imagery_stack[-1][None,...], imagery_stack, 
                                            imagery_stack[0][None,...]], axis = 0)
    

        masked_pixels = np.where(np.ma.getmaskarray(imagery_stack))

        if len(masked_pixels[0]) > 0:
            spatial_band_combs = np.stack([masked_pixels[i] for i in range(1,4)], axis = 0).T
            unique_spatial_band_combs = np.unique(spatial_band_combs, axis = 0)
            for ix in range(len(unique_spatial_band_combs)):
                timeseries = imagery_stack[:, unique_spatial_band_combs[ix, 0], unique_spatial_band_combs[ix, 1], 
                                          unique_spatial_band_combs[ix, 2]]

                valid_timesteps = np.argwhere(~np.ma.getmaskarray(timeseries)).flatten()
    

            # Interpolate if there are fewer valid timesteps than the overall number of timesteps
                f = interp1d(valid_timesteps, timeseries[valid_timesteps], kind='linear', fill_value='extrapolate')
                
                imagery_stack[:, unique_spatial_band_combs[ix, 0], unique_spatial_band_combs[ix, 1], 
                              unique_spatial_band_combs[ix, 2]] = f(range(imagery_stack.shape[0]))
        
        if self.pad_for_interpolation:
            imagery_stack = imagery_stack[1:-1]
        
        if self.smooth_temporally:
            imagery_stack = savgol_filter(imagery_stack, window_length=5, polyorder=3, axis=0)

        return np.array(imagery_stack)
            
    
    def return_srtm_layer(self):
        srtm_scenes, ctx = dl.scenes.search(
            self.dltile,
            products='srtm:GL1003',
            start_datetime='1999-12-30',
            end_datetime='2000-01-02',
            limit=None,
        )
        
        srtm_layer = srtm_scenes.mosaic(bands = 'slope', ctx=self.dltile)
        
        return np.array(srtm_layer)
    
    def find_chirps_imagery(self):
    
        chirps_scenes, ctx = dl.scenes.search(
            self.dltile,
            products='9a638ef860cf9d231775813e2b65241da41f576f:chirps_monthly_precipitation_tc',
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            limit=None)
        
        chirps_scenes = chirps_scenes.groupby('properties.date.year',
                                              'properties.date.month',
                                             )
        
        img_list = []
        for (year, month), month_scenes in chirps_scenes:
            # Use only a single CHIRPS value for the scene
            img = np.mean(month_scenes.mosaic(bands='monthly_precipitation', ctx = self.dltile))            
            img_list.append(img)
            
        return np.array(img_list)
    
    def find_lulc_imagery(self):
        '''
        Land-cover classification labels:
        0: water/nodata
        1: non-cropland
        2: cropland
        '''
    
        gfsad_scenes, ctx = dl.scenes.search(
                self.dltile,
                products='usgs:gfsad30:global:v1',
                start_datetime='2014-12-31',
                end_datetime='2015-01-31',
                limit=None)
        
        gfsad_layer = gfsad_scenes.mosaic(bands = 'Land_Cover', ctx = self.dltile)
        gfsad_layer = np.isin(gfsad_layer, [2]).astype(np.int)
    
        return gfsad_layer
    
    def generate_true_color_image(self, pred_folder):
    
        true_color_scenes, ctx = dl.scenes.search(
                self.dltile,
                products='sentinel-2:L1C',
                start_datetime='2019-12-15',
                end_datetime='2020-2-15',
                cloud_fraction=0.05, 
                limit = None)
        
        sorted_scenes = true_color_scenes.sorted(lambda s: s.properties.cloud_fraction, reverse=True)
        img = sorted_scenes.mosaic(bands='red green blue', ctx=self.dltile, processing_level='surface',
                                   scaling = [(0, 2500), (0, 2500), (0, 2500),])
        
        img = np.transpose(img, (1,2,0))
        
        return img
        
#         fig, ax = plt.subplots()
        
#         fontprops = fm.FontProperties(size=12)
#         bar_width = 50
#         scalebar = AnchoredSizeBar(ax.transData,
#                                            bar_width, '500m', 'lower right',
#                                            pad=0.3,
#                                            color='Black',
#                                            frameon=True,
#                                            size_vertical=2,
#                                            fontproperties=fontprops)

#         ax.add_artist(scalebar)
#         ax.imshow(img)
        
        
    
#         plt.savefig(f'{pred_folder}/rgb_{self.dltile.key}.png', bbox_inches='tight', pad_inches=0)