import numpy as np
import logging; logging.getLogger().setLevel(logging.INFO); logging.captureWarnings(True)
import sys, os
from glob import glob
import descarteslabs as dl
from descarteslabs.scenes import SceneCollection
import random
import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from dateutil.rrule import rrule, MONTHLY


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class RawImageryGenerator():

    def __init__(self, args, dltile):
        
        args = dotdict(vars(args))
        
        self.args = args
        
        self.start_date = self.args.start_date
        self.end_date = args.end_date
        self.dltile = dltile
        self.max_cloud_frac = args.max_cloud_frac
        self.month_valid_start_day = args.month_valid_start_day
        self.month_valid_end_day = args.month_valid_end_day
        
        self.pad_for_interpolation = True
        self.smooth_temporally = True
        
    def find_sentinel_imagery(self):
        print(f'Loading imagery between {self.start_date} and {self.end_date}')
        
        # Collect Sentinel-2 scenes
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
        
        # Filter scenes by year and month
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
        
        # Create list to store images
        imagery_list = []
    
        # Create datetime objects for start and end dates
        dt_strt = datetime.datetime.strptime(self.args.start_date, '%Y-%m-%d')
        dt_end = datetime.datetime.strptime(self.args.end_date, '%Y-%m-%d')
        
        # Create list of date tuples
        date_tuples = [(dt.year, dt.month) for dt in rrule(MONTHLY, 
                                                           dtstart=dt_strt, 
                                                           until=dt_end)]
        
        # Create empty masked array for imagery storing
        masked_array = np.ma.masked_all((len(date_tuples), self.dltile.tilesize, 
                                         self.dltile.tilesize, 2))    
            
        print('Collect raw Sentinel-2 imagery')
        for  dt_tuple, month_scenes in s2_scenes:
            
            # Extract month and year
            (year, month) = dt_tuple            
            month_index = np.argwhere([i == dt_tuple for i in date_tuples])[0][0]
  
            # Sort scenes (not strictly necessary)
            sorted_scenes = month_scenes.sorted(lambda s: s.properties.cloud_fraction,
                                               reverse=True)
            
            # Stack imagery
            stack = sorted_scenes.stack(bands = 'derived:evi derived:ndwi cloud-mask', 
                                          ctx = self.dltile,
                                         bands_axis=-1)
            
            # Extract derived layers + normalize
            stack_bands = (stack[..., 0:2] - 32767.5)/32767.5
            
            # Normalize
            stack_clouds = np.max(stack[..., 2:3], axis=-1, keepdims = True)
            
            # Tile cloud mask
            stack_clouds_tiled = np.tile(stack_clouds, (1,1,1, stack_bands.shape[-1]))
            masked_stack = np.ma.array(stack_bands, mask = stack_clouds_tiled)
            masked_stack_median = np.ma.median(masked_stack, axis = 0)
                        
            # Fill masked array    
            masked_array[month_index] = masked_stack_median
                 
        ## masked_array has size (36,256,256,2)        
        return masked_array 


    def temporal_interp_and_smoothing(self, imagery_stack):
        print('Temporal interpolation and smoothing')
        if self.pad_for_interpolation:
            # Pad first and last months for interpolation
            imagery_stack = np.ma.concatenate([imagery_stack[-1][None,...], imagery_stack, 
                                            imagery_stack[0][None,...]], axis = 0)
    
        # Find masked pixels
        masked_pixels = np.where(np.ma.getmaskarray(imagery_stack))

        if len(masked_pixels[0]) > 0:
            # Find unique spatial locations of masked pixels
            spatial_band_combs = np.stack([masked_pixels[i] for i in range(1,4)], axis = 0).T
            unique_spatial_band_combs = np.unique(spatial_band_combs, axis = 0)
            for ix in range(len(unique_spatial_band_combs)):
                # Extract timeseries at these masked pixel locations
                timeseries = imagery_stack[:, unique_spatial_band_combs[ix, 0], 
                                           unique_spatial_band_combs[ix, 1], 
                                          unique_spatial_band_combs[ix, 2]]
    
                # Extract timestpes where these masked values are
                valid_timesteps = np.argwhere(~np.ma.getmaskarray(timeseries)).flatten()
    

                # Define interpolation function
                f = interp1d(valid_timesteps, timeseries[valid_timesteps], kind='linear', 
                             fill_value='extrapolate')
                
                # Assign interpolated timeseries to pixel location
                imagery_stack[:, unique_spatial_band_combs[ix, 0], unique_spatial_band_combs[ix, 1], 
                              unique_spatial_band_combs[ix, 2]] = f(range(imagery_stack.shape[0]))
        
        if self.pad_for_interpolation:
            # Remove padded months 
            imagery_stack = imagery_stack[1:-1]
        
        if self.smooth_temporally:
            # Smooth temporal timeseries with a Savgol filter 
            imagery_stack = savgol_filter(imagery_stack, window_length=5, polyorder=3, axis=0)

        return np.array(imagery_stack)
            
    
    def return_srtm_layer(self):
        # Use SRTM layer to determine slope
        srtm_scenes, ctx = dl.scenes.search(
            self.dltile,
            products='srtm:GL1003',
            start_datetime='1999-12-30',
            end_datetime='2000-01-02',
            limit=None,
        )
        
        srtm_layer = srtm_scenes.mosaic(bands='slope', ctx=self.dltile)
        
        return np.array(srtm_layer)
    
    def find_chirps_imagery(self):
        # Extract CHIRPS imagery from user-defined DL catalog product
        chirps_scenes, ctx = dl.scenes.search(
            self.dltile,
            products='9a638ef860cf9d231775813e2b65241da41f576f:chirps_monthly_precipitation_tc',
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            limit=None)
        
        # Group by months
        chirps_scenes = chirps_scenes.groupby('properties.date.year',
                                              'properties.date.month',
                                             )
        
        img_list = []
        # Mosaic the images (shouldn't be an issue of multiple images per month, but just in case)
        for (year, month), month_scenes in chirps_scenes:
            # Use only a single CHIRPS value for the scene
            img = np.mean(month_scenes.mosaic(bands='monthly_precipitation', 
                                              ctx = self.dltile))            
            img_list.append(img)
            
        return np.array(img_list)
    
    def find_lulc_imagery(self):
        '''
        Land-cover classification labels:
        0: water/nodata
        1: non-cropland
        2: cropland
        '''
    
        # Extract GFSAD scenes
        gfsad_scenes, ctx = dl.scenes.search(
                self.dltile,
                products='usgs:gfsad30:global:v1',
                start_datetime='2014-12-31',
                end_datetime='2015-01-31',
                limit=None)
        
        # Mosaic and return pixels that are designated as cropland
        gfsad_layer = gfsad_scenes.mosaic(bands = 'Land_Cover', ctx = self.dltile)
        gfsad_layer = np.isin(gfsad_layer, [2]).astype(np.int)
    
        return gfsad_layer
    
    def generate_true_color_image(self, pred_folder):
        # Take recent RGB image from the winter/dry months
        true_color_scenes, ctx = dl.scenes.search(
                self.dltile,
                products='sentinel-2:L1C',
                start_datetime='2019-12-15',
                end_datetime='2020-2-15',
                cloud_fraction=0.05, 
                limit = None)
        
        # Mosaic and scale imagery
        sorted_scenes = true_color_scenes.sorted(lambda s: s.properties.cloud_fraction, 
                                                 reverse=True)
        img = sorted_scenes.mosaic(bands='red green blue', ctx=self.dltile, 
                                   processing_level='surface',
                                   scaling = [(0, 2500), (0, 2500), (0, 2500),])
        
        img = np.transpose(img, (1,2,0))
        
        return img
        