import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import clean_raw_data as clean

def year0(ttc_csv,
          old_poly_feats,
          new_poly_feats,
          imagery_dir,
          canopy_thresh,
          cloud_thresh,
          img_count,
          ):
    '''
    Assigns a remote or field verification method to every
    row in the input df, according baseline conditions (year0).
    Checks all rows have been assigned and prints method breakdown.
    Modifiable parameters include: 
        image requirements: Within 1 yr of planting date, there are 
        >=1 images available with <50% cloud cover
    
    Branch 1: at baseline, what is the canopy cover?
    Branch 2: what is the target land use?
    Branch 3: at baseline, how many images are available?

    TODO: determine how to handle polys w/ multiple target_sys
    '''
    ## Overview ##
    print('Running decision tree for Year 0: Baseline')
    print("------- DATA PREP ------- \n")
    print('Preparing inputs according to params...')
    # here would create the features according to input params (ie. image availability, slope, aspect)
    # xyz = img.imagery_features(imagery_dir,
    #                            new_poly_feats,
    #                            cloud_thresh)
    df = clean.clean_combine_inputs(ttc_csv, old_poly_feats, new_poly_feats, canopy_thresh)
    
    print("------- PORTFOLIO REPORT ------- \n")
    print(f"Unique project counts: {df['project_name'].nunique()}")
    print(f"Unique site counts: {df['site_name'].nunique()}")
    print(f"Unique polygon counts: {df['poly_name'].nunique()}")
    print(' ')
    print("Tree Cover Distribution (polygon):")
    print(f"Total open: {len(df[df.tree_cover <= canopy_thresh])}")
    print(f"Total closed: {len(df[df.tree_cover > canopy_thresh])} \n")
    print(' ')  
    print("Tree Cover Distribution (project):")
    print(f"Total open: {len(df[df.canopy == 'open'])}")
    print(f"Total closed: {len(df[df.canopy == 'closed'])} \n")

    ## BRANCH: open or closed ##
    open_ = df[df.canopy == 'open']
    closed_ = df[df.canopy == 'closed']

    ## BRANCH:

    ## BRANCH: image availability ##
    def image_availability(row, img_count):
        method = 'field' if row.baseline_img < img_count else 'remote'
        return method

    open_ = open_.assign(method=open_.apply(lambda row: image_availability(row, img_count), axis=1))
    closed_['method'] = 'field'
    
    comb = pd.concat([open_, closed_], ignore_index=True)
    assert not comb['method'].isnull().any(), "Some rows have not been assigned a method."
    print(f"All {len(comb)} polygons have been assigned a method.")
    method_counts = comb['method'].value_counts()
    method_breakdown = comb['method'].value_counts(normalize=True) * 100

    print("\nMethod allocation:")
    for method, count in method_counts.items():
        percentage = method_breakdown[method]
        print(f"{method}: {count} projects ({percentage:.2f}%)")
    
    print("\nSaving results to csv...")
    comb.to_csv("../data/results/decisons_yr0.csv", index=False)
    return comb

def year3(df):
    '''
    Assigns a remote or field verification method to every
    row in the input df, according endline conditions (year3).
    Checks all rows have been assigned and prints method breakdown.

    Modifiable parameters include: 
        image requirements: Within 1 yr of planting date, there are 
        >=1 images available with <50% cloud cover
    
    Branch 1: at midpoint, what is the canopy cover? 
    Branch 2: what is the target land use?
    Branch 3: at midpoint, how many images are available?

    TODO: determine how to handle polys w/ multiple target_sys
    '''

    return None

def year6(df):
    '''
    Assigns a remote or field verification method to every
    row in the input df, according endline conditions (year6).
    Checks all rows have been assigned and prints method breakdown.
    
    Modifiable parameters include: 
        image requirements: Within 1 yr of planting date, there are 
        >=1 images available with <50% cloud cover
    
    Branch 1: at endline, what is the canopy cover? There will only be 2
    landcover classes that could have <40% cover at year 6
    Branch 2: what is the target land use?
    Branch 3: at endline, how many images are available?

    TODO: determine how to handle polys w/ multiple target_sys
    '''

    return None