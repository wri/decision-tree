import pandas as pd
import geopandas as gpd
import numpy as np
import os
import process_api_results as clean

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
    img_df = img_df[['project_id', 'poly_id', 'site_id', 'datetime', 'eo:cloud_cover']].copy()
   
    # Ensure datetime format
    img_df['img_date'] = pd.to_datetime(img_df['datetime'], errors='coerce').dt.tz_localize(None)
    proj_df['plantstart'] = pd.to_datetime(proj_df['plantstart'], errors='coerce')
    proj_df['plantend'] = pd.to_datetime(proj_df['plantend'], errors='coerce')
    merged = img_df.merge(proj_df, on=['project_id', 'poly_id'], how='left')

    # Compute image availability window
    merged['date_diff'] = (merged['img_date'] - merged['plantstart']).dt.days

    baseline = merged[
        (merged['date_diff'] >= baseline_range[0]) &
        (merged['date_diff'] <= baseline_range[1]) &
        (merged['eo:cloud_cover'] < cloud_thresh)
    ]
    baseline_summary = (
        baseline.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='baseline_img_count')
    )

    ev = merged[
        (merged['date_diff'] >= ev_range[0]) &
        (merged['date_diff'] <= ev_range[1]) &
        (merged['eo:cloud_cover'] < cloud_thresh)
    ]
    ev_summary = (
        ev.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='ev_img_count')
    )
    final_summary = proj_df.merge(baseline_summary, on=['project_id', 'poly_id'], how='left') \
                           .merge(ev_summary, on=['project_id', 'poly_id'], how='left')

    # Fill in missing image counts with 0
    final_summary[['baseline_img_count', 'ev_img_count']] = final_summary[
        ['baseline_img_count', 'ev_img_count']
    ].fillna(0)

    return final_summary