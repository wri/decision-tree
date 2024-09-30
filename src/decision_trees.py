import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

def year0(df):
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
    print(f"Unique project counts: {df['project'].nunique()}")
    print(f"Unique site counts: {df['sitename'].nunique()}")
    print(f"Unique polygon counts: {df['poly_name'].nunique()}")

    ## BRANCH 1 ##
    open_ = df[df.canopy == 'open']
    closed_ = df[df.canopy == 'closed']
            
    # ## BRANCH 2 ##
    # # not represented: wetland,plantation / urban-forest
    # open_landcovers = ['agroforest', 
    #                    'mangrove', 
    #                    'wetland', 
    #                    'silvopasture', 
    #                    'plantation', 
    #                    'natural-forest', 
    #                    'agroforest,silvopasture',
    #                    'agroforest,wetland',
    #                    ]
    # closed_landcovers = ['plantation', 
    #                      'natural-forest',
    #                      'urban-forest'
    #                      ]
    # open_sys = open_[open_.target_sys.isin(open_landcovers)]
    # closed_sys = closed_[closed_.target_sys.isin(closed_landcovers)]

    ## BRANCH 3 ##
    def image_availability(row):
        method = 'field' if row.baseline_img < 1 else 'remote'
        return method
    open_ = open_.assign(method=open_.apply(image_availability, axis=1))
    closed_['method'] = 'field'
    
    comb = pd.concat([open_, closed_], ignore_index=True)
    print(comb.shape)
    assert not comb['method'].isnull().any(), "Some rows have not been assigned a method."
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