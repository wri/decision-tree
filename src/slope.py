import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
        
def apply_slope_classification(params, df, slope_stats):
    '''
    Classify polygons based on the % of the area that is steep (> threshold slope).
    
    - "steep" if the % of area > threshold.
    - "flat" if the % of area <= threshold.
    - NaN if slope_area is NaN (no data).
    
    Applies to all polygons, but downstream decision tree will filter for "remote" rows.
    '''
    n_projects = slope_stats['project_id'].nunique() 
    n_polys = slope_stats['poly_id'].nunique() 
    print(f"Analyzing slope for {n_projects} projects and {n_polys} polygons...")

    slope_thresh = params['criteria']['slope_thresh']
    slope_stats = slope_stats[['project_id', 'poly_id', 'slope_area']].copy()

    def classify_slope(val):
        if pd.isna(val):
            return 'missing'
        elif val > slope_thresh:
            return 'steep'
        else:
            return 'flat'

    # Apply classification
    slope_stats.loc[:, 'slope'] = slope_stats['slope_area'].apply(classify_slope)

    # Merge into main DataFrame
    comb = df.merge(
        slope_stats[['project_id', 'poly_id', 'slope_area', 'slope']], 
        on=['project_id', 'poly_id'], 
        how='left' 
    )
    
    num_duplicates = comb.duplicated().sum()
    print(f"DUPLICATES: {num_duplicates}")   
    
    return comb


