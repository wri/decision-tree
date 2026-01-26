import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import math


def summarize_results(df):
    total_projects = df['project_id'].nunique()
    print(f"{total_projects} total projects")
    
    # 2. Total number of polygon_ids per project
    polygon_counts = df.groupby('project_id')['poly_id'].nunique().reset_index(name='polygon_count')
    
    # 3. Proportion of remote vs field decisions per project
    decision_counts = df.groupby(['project_id', 'decision']).size().unstack(fill_value=0)    
    decision_proportions = decision_counts.div(decision_counts.sum(axis=1), axis=0)
    decision_proportions = (decision_proportions * 100).round(2).reset_index()
    
    # Merge polygon counts and decision proportions into one summary
    summary = polygon_counts.merge(decision_proportions, on='project_id')
    return summary


### Cleans the tropical tree cover statistics for polygons ###

def classify_canopy(project: pd.DataFrame, thresh: int):
    """
    Determines the canopy classification ('open', 'closed', or 'unknown') 
    for a project.
    
    Parameters:
    - project: DataFrame containing rows for a single project.
    - thresh: Tree cover threshold for determining open or closed canopy.
    
    Returns:
    - A string: 'open', 'closed', or 'unknown'.
    """
    closed_canopy = (project['tree_cover'] > thresh).sum()
    open_canopy = (project['tree_cover'] <= thresh).sum()

    if open_canopy > closed_canopy:
        return 'open'
    elif closed_canopy > open_canopy:
        return 'closed'
    else:
        return 'unknown'

def clean_ttc_csv(csv: str, canopy_thresh: int):
    
    '''
    Performs cleaning steps on tree cover stats.
    "Open" or "closed" canopy is defined at the project scale. Some polygons may meet 
    the closed definition, but if the majority of polygons in a project are open canopy,
    it will be classified as open. Each project will have a single designation.
    '''

    ttc_raw = pd.read_csv(csv)
    ttc = ttc_raw[['tree_cover', 'project_name', 'site_name', 'poly_name']]
    print(f"Dropping {len(ttc[ttc['tree_cover'].isna()])} rows with null TTC values.\n")
    ttc[ttc.tree_cover.isna()].to_csv('../data/qaqc/ttc_nulls.csv')
    ttc = ttc.dropna(subset=['tree_cover'])

    ttc['tree_cover'] = ttc['tree_cover'].round()
    assert ttc['tree_cover'].min() >= 0.0, f"Invalid min tree_cover: {ttc['tree_cover'].min()}"
    assert ttc['tree_cover'].max() <= 100.0, f"Invalid max tree_cover: {ttc['tree_cover'].max()}"

    def apply_canopy_classification(project_df):
        canopy_classification = classify_canopy(project_df, canopy_thresh)
        project_df['canopy'] = canopy_classification
        return project_df

    ttc = ttc.groupby('project_name').apply(apply_canopy_classification)
    ttc = ttc.reset_index(drop=True)
    assert ttc.groupby('project_name')['canopy'].nunique().max() == 1, \
        "Each project must have a single canopy designation."

    return ttc


### Cleans the polygon features csv ###

def clean_polygon_csv(original_csv, new_csv):
    '''
    Merges shp and ft_poly to use the updated target sys and practice values
    to create a single csv file with polyon features
    Performs cleaning steps and returns comb df
    '''
    shp = gpd.read_file(new_csv)
    ft_poly = pd.read_csv(original_csv)

    ft_poly.rename(columns={'target_sys':'target_sys_og',
                             'practice':'practice_og'}, inplace=True)
    ft_poly_new = pd.merge(ft_poly, 
                        shp[['Project', 'SiteName', 'poly_name','target_sys', 'practice']], 
                        how='inner', 
                        on=['Project', 'SiteName', 'poly_name']).drop_duplicates()

    # Identify all duplicates, including the first occurrence, then
    # count and drop, keeping the first occurence
    columns_to_match = ['Project', 'SiteName', 'poly_name', 'plantstart', 'Area_ha']
    ft_poly_new_cleaned = ft_poly_new.drop_duplicates(subset=columns_to_match)
    rows_dropped = len(ft_poly_new) - len(ft_poly_new_cleaned)
    print(f"Dropping {rows_dropped} duplicate polygon rows.\n")
    ft_poly_new_cleaned.columns = ft_poly_new_cleaned.columns.map(lambda x: re.sub(' ', '_', x.lower().strip()))
    feats = ft_poly_new.rename(columns={'available_baseline_images': 'baseline_img',
                                        'available_ev_images':'ev_img',
                                        'Project': 'project_name',
                                        'SiteName': 'site_name'
                                        })
    # manually clean target sys and intervention columns
    feats['target_sys'] = feats['target_sys'].str.replace(r'\briparian-area-or-wetland\b', 'wetland', regex=True)
    feats['target_sys'] = feats['target_sys'].str.replace(r'\briparian-or-wetland\b', 'wetland', regex=True)
    feats['target_sys'] = feats['target_sys'].str.replace(r'\bwoodlot-or-plantation\b', 'plantation', regex=True)

    feats['practice'] = feats['practice'].replace({
        'direct-seedling':'direct-seeding',
        'direct-seeding,tree-planting':'tree-planting,direct-seeding',
        'assisted-natural-regeneration,tree-planting':'tree-planting,assisted-natural-regeneration',
        'assisted-natural-regeneration,direct-seeding':'direct-seeding,assisted-natural-regeneration',
    })
    feats['practice'] = feats['practice'].str.replace(r'\bassisted-natural-regeneration\b', 'ANR', regex=True)

    # drop target systems that are null
    null_sys = feats[feats['target_sys'].isna()]
    print(f"Dropping {len(null_sys)} rows with NaN target system.\n")
    feats = feats.dropna(subset=['target_sys'])
    lcs = list(set(feats.target_sys))
    print(f"{len(lcs)} target systems: {lcs}\n")
    return feats

def clean_combine_inputs(ttc_csv, 
                         original_feats_csv, 
                         new_feats_csv,
                         canopy_thresh,
                         outpath='../data/decision_tree_data_clean.csv'):
    '''
    cleans ttc and polygon csvs
    '''

    ttc = clean_ttc_csv(ttc_csv, canopy_thresh)
    feats = clean_polygon_csv(original_feats_csv, new_feats_csv)
    comb = pd.merge(feats, ttc, on=['project_name', 'site_name', 'poly_name']) ## empty dataframe
    comb.to_csv(outpath)
    print(f"Cleaned input data saved at {outpath}")
    return ttc, feats