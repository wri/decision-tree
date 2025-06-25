import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

def apply_slope_classification(params, projects, slope_stats):
    '''
    Classify polygons based on the % of the area that is steep (>20% slope).
    
    - "steep" if the % of area > threshold.
    - "flat" if the % of area <= threshold.
    - "unknown" if area_stat is NaN (no data)
    Applies to all polygons, but downstream decision tree will filter for "remote" rows.
    '''
    slope_thresh = params['criteria']['slope_thresh']
    slope_stats = slope_stats[['project_id', 'poly_id', 'slope_area']]
    slope_stats['slope'] = np.where(slope_stats['slope_area'].isna(), 'unknown',
                                    np.where(slope_stats['slope_area'] > slope_thresh,'steep','flat'
                                    )
                                )
    comb = projects.merge(
        slope_stats[['project_id', 'poly_id', 'slope_area', 'slope']], 
        on=['project_id', 'poly_id'], 
        how='left'
    )
    return comb


