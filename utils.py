import numpy as np
from rasterio import Affine
import rasterio
from scipy.ndimage import convolve

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
    
