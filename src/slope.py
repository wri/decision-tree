import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

def apply_slope_classification(df, param_path):
    '''
    Classify polygons based on the % of the area that is steep (>20% slope).
    
    - "steep" if the % of area > threshold.
    - "flat" if the % of area <= threshold.
    - "unknown" if area_stat is NaN.
    Applies to all polygons, but downstream decision tree will filter for "remote" rows.
    '''
    with open(param_path) as file:
        params = yaml.safe_load(file)
    
    criteria = params.get('criteria', {})
    outfile = params.get('outfile', {})
    slope_thresh = criteria.get('slope_thresh')  
    slope_csv = outfile.get('slope')
    slope_csv = '../data/slope/slope_percentage.csv' #remove
    slope_stats = pd.read_csv(slope_csv)
    slope_stats = slope_stats[['project_id', 'poly_id', 'slope_area']]
    slope_stats['slope'] = np.where(slope_stats['slope_area'].isna(), 'unknown',
                                    np.where(slope_stats['slope_area'] > slope_thresh,'steep','flat'
                                    )
                                )
    comb = df.merge(
        slope_stats[['project_id', 'poly_id', 'slope_area', 'slope']], 
        on=['project_id', 'poly_id'], 
        how='left'
    )
    return comb


