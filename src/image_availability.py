import pandas as pd
import geopandas as gpd
import numpy as np
import os

def analyze_image_availability(proj_df, 
                               img_df, 
                               baseline_range=(-365, 0), 
                               ev_range=(365, 1095), 
                               cloud_thresh=50):
    """
    Assesses image availability for baseline & early verification per 
    project/polygon based on user defined windows.

    Parameters:
    - proj_df (pd.DataFrame): DataFrame containing project characteristics.
    - img_df (pd.DataFrame): DataFrame containing image observations.
    - baseline_range (tuple): Timeframe for baseline availability in days 
        (e.g., (-365, 0) for 1 year before plantstart).
    - ev_range (tuple): Timeframe for early verification availability in days 
        (e.g., (90, 1095) for 3 years after plantstart).
    - cloud_thresh (int): Maximum acceptable cloud cover percentage.

    Returns:
    - pd.DataFrame: Merged DataFrame with image availability counts per polygon.
    """
    proj_df.columns = proj_df.columns.str.lower()
    img_df = img_df[['project_id', 'project_na', 'poly_id', 'datetime', 'eo:cloud_cover']].copy()

    # Convert 'datetime' columns
    img_df.loc[:, 'datetime'] = pd.to_datetime(img_df['datetime'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    img_df.loc[:, 'datetime'] = img_df['datetime'].apply(lambda x: x.replace(tzinfo=None) if pd.notna(x) else x)
    img_df = img_df.rename(columns={'datetime': 'img_date'})

    # Merge project data with image data
    new = img_df.merge(proj_df, on=['project_id', 'poly_id'], how='left')
    new['plantstart'] = pd.to_datetime(new['plantstart'], errors='coerce')

    # Calculate img availability
    new['date_diff'] = (new['img_date'] - new['plantstart']).dt.days
    baseline = new[
        (new['date_diff'] >= baseline_range[0]) &  
        (new['date_diff'] <= baseline_range[1]) &  
        (new['eo:cloud_cover'] < cloud_thresh)  
    ]
    baseline_summary = (
        baseline.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='baseline_img_count')
    )

    ev = new[
        (new['date_diff'] >= ev_range[0]) & 
        (new['date_diff'] <= ev_range[1]) &  
        (new['eo:cloud_cover'] < cloud_thresh)
    ]

    ev_summary = (
        ev.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='ev_img_count')
    )

    # **Merge baseline and early verification counts into the final summary**
    final_summary = proj_df.merge(baseline_summary, on=['project_id', 'poly_id'], how='left') \
                           .merge(ev_summary, on=['project_id', 'poly_id'], how='left')

    # Fill NaN values with 0 (if no images met the criteria)
    final_summary[['baseline_img_count', 'ev_img_count']] = final_summary[['baseline_img_count', 
                                                                           'ev_img_count']].fillna(0)

    return final_summary