# dl_irrigation_prediction
Predicting irrigation presence using the Descartes Labs platform.

## Overview
This repository contains the code necesssary to deploy a pretrained binary classification model to predict irrigation presence. The classification model uses layers derived from Sentinel-2 imagery and CHIRPS rainfall predictions, SRTM DEM measurements, and GFSAS land cover classifications for prediction. After basic morphological cleaning, the predictions are plotted, polygonized, and saved as KML files (for visual inspection in Google Earth Pro). 

The repository consists of 4 scripts and a parameters file. These files are as follows:

1. `main.py`: This script collects the pretrained model, necessary imagery, and paramerization; it then makes predictions over a spatial extent defined by a DLTile. The script also contains the prediction output functions, which organize the predictions into a series of matplotlib plots and a KML shapefile. Run this file to make the irrigation predictions.
2. `params.yaml`: This file contains parameters for model prediction, including the desired imagery date range, the pretrained model name, and the maximum allowable slope for prediction. 
3. `prediction_layer_generator.py`: This script contains a custom object that creates and returns the desired feature layers for prediction for the desired spatial extent.  
4. `sentinel_imagery_generator.py`: This script contains a custom object that contains functions for returning all imagery required for the predictions.
5. `utils.py`: This script contains general utility functions, including those for prediction dilation, erosion, and polygonization. 

Before running the code contained in this repository via `python main.py`, users must also create the following three folders and subdirectories (or rename all paths contained in the scripts):

```
dl_irrigation_project
|--- data
|    |--- processed_data
|    |    |--- evi_max_min_ratio_layers
|    |    |--- feature_layers
|    |--- raw_data
|         |--- imagery_stacks
|--- predictions
|--- pretrained_models
     |--- models
     |--- normalization_arrays
```

## Pretrained models and normalizations

Pretrained models and normalization files are available at the following public [Google
 storage bucket](https://console.cloud.google.com/storage/browser/qsel_irrigation_detection) in the `pretrained_model_files/` folder. Download a pretrained model and normalization array from the bucket and save to the `pretrained_models/` folder in this repository to make predictions with an existing model. 