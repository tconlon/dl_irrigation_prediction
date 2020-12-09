import numpy as np
import sys, os
import geopandas as gpd
from tqdm import tqdm
from glob import glob

import descarteslabs as dl
from descarteslabs.scenes import SceneCollection
import random
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
# from pysptools.abundance_maps.amaps import UCLS, NNLS, FCLS


class FeatureLayerGenerator():

    def __init__(self, args, sentinel_imagery_stack, chirps_array, srtm_layer, gfsad_layer):
        
        self.args = args
        
        # Define which features are calculated for the prediction
        self.srtm = True

        ## EVI-based metrics
        self.evi_annual_corrcoef = True

        self.evi_chirps_corrcoef = True
        self.evi_chirps_rolling_corrcoef_perc = []
        
        self.evi_at_min_n_chirps = [6, 12, 18] # Half of the MODIS values
        
        self.evi_max_min_ratio_maxval = [95, 90, 85, 80]
        
        ## NDWI-based metrics
        self.ndwi_annual_corrcoef = True

        self.ndwi_chirps_corrcoef = True
        self.ndwi_chirps_rolling_corrcoef_perc = [1,5,10,25,75,90,95,99]
        
        self.ndwi_at_min_n_chirps = [6, 12, 18] # Half of the MODIS values
        
        # Create an AMAP
        self.evi_amap = False
        
        ## Define imagery layers
        self.evi_imagery_stack = sentinel_imagery_stack[..., 0]        
        self.ndwi_imagery_stack = sentinel_imagery_stack[..., 1]        
        self.chirps_array = chirps_array
        self.srtm_layer = srtm_layer
        self.gfsad_layer = gfsad_layer
        
        self.rows = sentinel_imagery_stack.shape[1]
        self.cols = sentinel_imagery_stack.shape[2]
        
        self.window_size = 5
        
        self.feature_list = []
        
    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
     
    def annual_corrcoef_func(self, imagery_stack, layer_name):
        print('Create annual correlation coefficient layer')
        annual_corrcoef_layer = np.zeros((1, self.rows, self.cols))
        for row in tqdm(range(self.rows)):
            for col in range(self.cols):
                y1 = imagery_stack[0:12,  row, col]
                y2 = imagery_stack[12:24, row, col]
                y3 = imagery_stack[24:36, row, col]
                    
                annual_correlation_matrix = np.corrcoef(np.array([y1, y2, y3]))
                annual_corrcoef_layer[0, row, col] = np.mean([annual_correlation_matrix[1,0],
                                                             annual_correlation_matrix[2,0],
                                                             annual_correlation_matrix[2,1]])
        self.feature_list.append(annual_corrcoef_layer)    

    def imagery_chirps_corrcoef_func(self, imagery_stack, layer_name):
        print('Create imagery-CHIRPS correlation coefficient layer')
        imagery_chirps_corrcoef_layer = np.zeros((1, self.rows, self.cols))
        
        if layer_name == 'evi':
            chirps_array = np.hstack([self.chirps_array[-1], self.chirps_array[0:-1]])
        elif layer_name == 'ndwi':
            chirps_array = self.chirps_array
        
        for row in tqdm(range(self.rows)):
            for col in range(self.cols):
                imagery_chirps_coeff = np.corrcoef(imagery_stack[:,row, col],
                                                   chirps_array)
                imagery_chirps_corrcoef_layer[0, row, col] = imagery_chirps_coeff[0][1]
                    
        self.feature_list.append(imagery_chirps_corrcoef_layer)
    
    def imagery_chirps_rolling_corrcoef_func(self, imagery_stack, percentiles, layer_name):
        print('Create imagery-CHIRPS rolling correlation coefficient layer')
        imagery_chirps_rolling_corrcoef_layer = \
        np.zeros((imagery_stack.shape[0] - self.window_size + 1, self.rows, self.cols))
        
        if layer_name == 'evi':
            chirps_array = np.hstack([self.chirps_array[-1], self.chirps_array[0:-1]])
        elif layer_name == 'ndwi':
            chirps_array = self.chirps_array
            
        chirps_rolling = self.rolling_window(chirps_array, self.window_size)

        for row in tqdm(range(self.rows)):
            for col in range(self.cols):
                a = imagery_stack[:,row, col]
                a_rolling = self.rolling_window(a, self.window_size)
                rolling_coeffs = np.zeros((a_rolling.shape[0]))

                for ix in range(a_rolling.shape[0]):
                    rolling_coeffs[ix] = np.corrcoef(a_rolling[ix], chirps_rolling[ix])[0][1]
                    
                imagery_chirps_rolling_corrcoef_layer[:, row, col] = rolling_coeffs
                    
        imagery_chirps_rolling_perc = np.percentile(imagery_chirps_rolling_corrcoef_layer, 
                                                    percentiles,
                                                    axis=0)
            
        self.feature_list.append(imagery_chirps_rolling_perc)
            
    def imagery_at_min_chirps_func(self, imagery_stack, min_n_chirps_vals, layer_name):
        print('Calculating imagery values at minimum CHIRPS values')
        imagery_at_min_chirps_layer_mean = np.zeros((len(min_n_chirps_vals), self.rows, self.cols))
        imagery_at_min_chirps_layer_max = np.zeros((len(min_n_chirps_vals), self.rows, self.cols))
        
        if layer_name == 'evi':
            chirps_array = np.hstack([self.chirps_array[-1], self.chirps_array[0:-1]])
        elif layer_name == 'ndwi':
            chirps_array = self.chirps_array
            
        sorted_indices = chirps_array.argsort()
        
        for row in tqdm(range(self.rows)):
            for col in range(self.cols):
                imagery_ts = imagery_stack[:,row, col]
                for ix, n in enumerate(min_n_chirps_vals):
                    imagery_at_min_chirps_layer_mean[ix, row, col] = np.mean(imagery_ts[sorted_indices[0:n]])
                    imagery_at_min_chirps_layer_max[ix, row, col] = np.max(imagery_ts[sorted_indices[0:n]])
                    
        ## Adjust to MODIS range
        
        imagery_at_min_chirps_layer_mean = imagery_at_min_chirps_layer_mean * 10000
        imagery_at_min_chirps_layer_max = imagery_at_min_chirps_layer_max * 10000
        
        self.feature_list.append(imagery_at_min_chirps_layer_mean)
        self.feature_list.append(imagery_at_min_chirps_layer_max)
        
    def imagery_max_min_ratio_func(self, imagery_stack, max_values, layer_name):
        print('Calculate max:min imagery ratio')
        imagery_max_min_ratio = np.zeros((len(max_values),  self.rows, self.cols))
        
        percentiles = max_values + [100 - i for i in max_values]
                                        
        imagery_percentiles = np.percentile(imagery_stack, percentiles, axis=0)                           
        
        for ix in range(len(max_values)):
            imagery_max_min_ratio[ix] = imagery_percentiles[ix]/imagery_percentiles[ix + len(max_values)]

        self.feature_list.append(imagery_max_min_ratio)    
            
        if layer_name == 'evi':
            return imagery_max_min_ratio
        
    def create_amap(self, imagery_stack, layer_name):
        
        # Adjust to MODIS range
        imagery_stack = imagery_stack*10000
        
        print(f'Creating abundance map for {layer_name}')
        evergreen_ts = np.ones(self.chirps_array.shape)
        tEMs = np.stack((self.chirps_array, evergreen_ts), axis=0)
        
        amap = np.zeros((4, self.rows, self.cols))
                
        for row in tqdm(range(self.rows)):
            for col in range(self.cols):
                imagery_ts = imagery_stack[:,row, col][None,...]
                tEMs_coeff = NNLS(imagery_ts, tEMs)
                
                ts_recreated = np.matmul(tEMs_coeff, tEMs)
                
                ts_diff = np.abs(imagery_ts - ts_recreated)
                mean_error  = np.expand_dims(np.mean(ts_diff, axis = -1), axis=-1)
                max_error = np.expand_dims(np.max(ts_diff, axis = -1), axis=-1)
                
                coeffs_for_output = np.concatenate((tEMs_coeff, mean_error, max_error), axis = -1)
                amap[:, row, col] = coeffs_for_output
    
        self.feature_list.append(amap) 
    
    def return_valid_pixels(self, evi_max_min_ratio):
        
        srtm_layer_valid = self.srtm_layer < self.args.srtm_max_slope
        evi_max_min_ratio_valid = evi_max_min_ratio[1] > self.args.min_evi_ratio_90_10
        lulc_valid = self.gfsad_layer
        
        valid_pixels = ((srtm_layer_valid * evi_max_min_ratio_valid * lulc_valid) > 0).astype(np.int)
        
        return valid_pixels
                            
        
    def create_features(self):
        
        if self.srtm:
            print('Appending SRTM layer')
            self.feature_list.append(self.srtm_layer)
            
        ## EVI feature creation
        if self.evi_annual_corrcoef:
            self.annual_corrcoef_func(self.evi_imagery_stack, layer_name='evi')
        
        if self.evi_chirps_corrcoef:
            self.imagery_chirps_corrcoef_func(self.evi_imagery_stack, layer_name='evi')
            
        if len(self.evi_at_min_n_chirps) > 0:
            self.imagery_at_min_chirps_func(self.evi_imagery_stack, self.evi_at_min_n_chirps, 
                                               layer_name='evi')
            
        if len(self.evi_max_min_ratio_maxval) > 0:
            evi_max_min_ratio = self.imagery_max_min_ratio_func(self.evi_imagery_stack, 
                                                                    self.evi_max_min_ratio_maxval,
                                                                    layer_name='evi')
        
        ## NDWI feature creation
        if self.ndwi_annual_corrcoef:
            self.annual_corrcoef_func(self.ndwi_imagery_stack, layer_name='ndwi')
    
        if self.ndwi_chirps_corrcoef:
            self.imagery_chirps_corrcoef_func(self.ndwi_imagery_stack, layer_name='ndwi')
            
    
        if len(self.ndwi_at_min_n_chirps) > 0:
            self.imagery_at_min_chirps_func(self.ndwi_imagery_stack, self.ndwi_at_min_n_chirps,
                                               layer_name='ndwi')
            
        if self.evi_amap:
            self.create_amap(self.evi_imagery_stack, layer_name='evi')
            
        # Concatenate layers
        feature_stack = np.concatenate(self.feature_list, axis = 0).astype(np.float32)
        
        return feature_stack,  evi_max_min_ratio
    
