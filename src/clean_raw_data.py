import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import math

### Cleans the tropical tree cover statistics for polygons ###

def classify_canopy(project, thresh):
    closed_canopy = (project['ttc'] > thresh).sum()
    open_canopy = (project['ttc'] <= thresh).sum()
    return 'open' if open_canopy > closed_canopy else 'closed'

def clean_ttc_csv(csv: str,
                  canopy_thresh: int):
    
    '''
    Performs cleaning steps on ttc statistics.
    TTC tree cover
    Canopy column is at the project scale. Some polygons might have closed ttc, 
    but if the majority of polygons in a project are open canopy, it will be
    classified as open.
    '''

    ttc_raw = pd.read_excel(csv)
    to_keep = ['Percent Tree Cover 2020', 'Project', 'SiteName', 'poly_name']
    ttc = ttc_raw[to_keep]
    ttc.columns = ttc.columns.map(lambda x: re.sub(' ', '_', x.lower().strip()))
    ttc = ttc.rename(columns={'percent_tree_cover_2020': 'ttc'})
    print(f"Dropping {len(ttc[ttc['ttc'] == 'TTC_NA'])} null values for TTC \n")

    ttc = ttc[ttc['ttc'] != 'TTC_NA']
    ttc['ttc'] = ttc['ttc'].astype('float')
    ttc['ttc'] = ttc['ttc'].round()
    
    assert ttc['ttc'].min() >= 0.0, print(ttc['ttc'].min())
    assert ttc['ttc'].max() <= 100.0, print(ttc['ttc'].max())

    print("Polygon Cover Distribution:")
    print(f"Total polys <={canopy_thresh}% cover: {len(ttc[ttc.ttc <= canopy_thresh])}")
    print(f"Total polys >{canopy_thresh}% cover: {len(ttc[ttc.ttc > canopy_thresh])} \n")

    # .transform() method in pandas is used to apply a function to a group or subset of a df and 
    # return an aligned result, meaning the output has the same shape as the input.
    ttc['canopy'] = ttc.groupby('project')['ttc'].transform(lambda x: classify_canopy(ttc[ttc['project'] == x.name], 
                                                                                      canopy_thresh
                                                                                      ))
    print("Project Cover Distribution:")
    print(ttc.canopy.value_counts())

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

    # around 2k rows are duplicates between colums_to_match
    # these can be dropped since we don't use slope/aspect yet.
    columns_to_match = ['Project', 'SiteName', 'poly_name', 'plantstart', 'Area_ha']
    dups = ft_poly_new.duplicated(subset=columns_to_match, keep=False)
    print(f"\n{dups.sum()} remaining duplicates will be dropped.")
    ft_poly_new = ft_poly_new.drop_duplicates(subset=columns_to_match)
    ft_poly_new.columns = ft_poly_new.columns.map(lambda x: re.sub(' ', '_', x.lower().strip()))
    feats = ft_poly_new.rename(columns={'available_baseline_images': 'baseline_img',
                                              'available_ev_images':'ev_img'})
    
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

    lcs = list(set(feats.target_sys))
    print(f"\n{len(lcs)} target systems:")
    print(lcs)
    
    return feats

def clean_combine_inputs(ttc_csv, 
                         canopy_thresh,
                         original_feats_csv, 
                         new_feats_csv,
                         outpath='../data/decision_tree_data_clean.csv'):

    ttc = clean_ttc_csv(ttc_csv, canopy_thresh)
    feats = clean_polygon_csv(original_feats_csv, new_feats_csv)

    comb = pd.merge(feats, ttc, on=['project', 'sitename', 'poly_name'])
    comb.to_csv(outpath)

    return comb