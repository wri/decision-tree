import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import clean_raw_data as clean

def parse_condition(value, actual):
    """
    Evaluate simple condition like '>=1', '<2', or direct value.
    Interprets and evaluates conditional expressions (like ">=1") 
    from the rules CSV.
    """
    if isinstance(value, str) and any(op in value for op in [">=", "<=", ">", "<", "=="]):
        try:
            return eval(f"{actual} {value}")
        except Exception:
            return False
    return actual == value

def apply_rules_baseline(df, save_to_csv=None):
    '''
    Refers to the rules template to determine:
    Branch 1: what is the canopy cover at baseline?
    Branch 2: what is the target land use?
    Branch 3: what is the practice?
    Branch 4: what is image availability at baseline?
    '''

    rules = pd.read_csv('../data/rule_template.csv')
    decisions = []

    for _, row in df.iterrows():
        matched = False
        for _, rule in rules.iterrows():
            if (
                row['baseline_canopy'] == rule['baseline_canopy'] and
                row['target_sys'] == rule['target_sys'] and
                row['practice'] == rule['practice'] and
                parse_condition(rule['img_count'], row['baseline_img_count'])
            ):
                decisions.append(rule['decision'])
                matched = True
                break
        if not matched:
            decisions.append('review required')

    df['decision'] = decisions

    # Summary output
    summary = df['decision'].value_counts().reset_index()
    summary.columns = ['decision', 'count']
    summary['proportion'] = summary['count'] / len(df)

    if save_to_csv:
        df.to_csv(save_to_csv, index=False)

    return df, summary

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