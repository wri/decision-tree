import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

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

# def apply_rules_baseline(params, df):
#     '''
#     Refers to the rules template to determine:
#     Branch 1: what is the canopy cover (at baseline)?
#     Branch 2: what is the target land use?
#     Branch 3: what is the practice?
#     Branch 4: what is image availability?
    
#     Outputs:
#         - strong remote
#         - weak remote
#         - strong field
#         - weak field
#     '''
#     n_projects = df['project_id'].nunique()
#     n_polys = df['poly_id'].nunique() 
#     print(f"Running the decision tree on {n_projects} projects and {n_polys} polygons...")

#     rules = pd.read_csv(params['criteria']['rules'])
#     rules = rules.map(lambda x: str(x).strip().lower() if isinstance(x, str) else x)
#     for col in ['baseline_canopy', 'target_sys', 'practice', 'slope']:
#         df[col] = df[col].astype(str).str.strip().str.lower()
#     # update missing to nans
#     #df['slope'] = df['slope'].apply(lambda x: 'missing' if pd.isna(x) else x)
#     decisions = []

#     for _, row in df.iterrows():
#         matched = False
#         for _, rule in rules.iterrows():
            
#             if (
#                 row['baseline_canopy'] == rule['baseline_canopy'] and
#                 row['target_sys'] == rule['target_sys'] and
#                 row['practice'] == rule['practice'] and
#                 parse_condition(rule['img_count'], row['baseline_img_count']) and
#                 (rule['slope'] == 'any' or row['slope'] == rule['slope'])
#             ):
#                 # Base decision from rules
#                 decisions.append(rule['decision'])
#                 matched = True
#                 break
#         if not matched:
#             decisions.append('review required')
#             continue

#     df['decision'] = decisions

#     # Optional: drop duplicate rows if needed
#     # df = df.drop_duplicates()

#     desired_cols = [
#     'project_id', 'poly_id', 'site_id', 'project_name',
#     'plantstart', 'practice', 'target_sys', 'baseline_year',
#     'ttc_2020', 'ttc_2021', 'ttc_2022', 'ttc_2023',  'ttc_2024', 
#     'baseline_img_count', 'ev_img_count', 'baseline_canopy', 'ev_canopy', 
#     'slope_area', 'slope', 'decision'
#     ]
#     df = df[desired_cols]

#     # Summary output
#     summary = df['decision'].value_counts().reset_index()
#     summary.columns = ['decision', 'count']
#     summary['proportion'] = summary['count'] / len(df)
#     print(summary)

#     return df

## NOt this
def apply_rules_baseline(params, df):
    '''
    Decision tree for baseline classification.
    
    PHASE 1:
    - Use rules template to classify as either remote or field
      (ignores slope entirely for remote vs. field)
    
    PHASE 2:
    - For rows assigned to field, split into strong or weak based on slope
    '''
    n_projects = df['project_id'].nunique()
    n_polys = df['poly_id'].nunique() 
    print(f"Running the decision tree on {n_projects} projects and {n_polys} polygons...")

    rules = pd.read_csv(params['criteria']['rules'])
    rules = rules.map(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

    for col in ['baseline_canopy', 'target_sys', 'practice', 'slope']:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # ensure slope exists as a string
    df['slope'] = df['slope'].fillna('missing')
    df['slope'] = df['slope'].astype(str).str.strip().str.lower()

    decisions = []

    for _, row in df.iterrows():
        matched = False
        decision = None

        # PHASE 1: match remote/field ignoring slope
        for _, rule in rules.iterrows():
            
            if rule['decision'] in ['strong remote', 'weak remote']:
                if (
                    row['baseline_canopy'] == rule['baseline_canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    parse_condition(rule['img_count'], row['baseline_img_count'])
                ):
                    decision = rule['decision']
                    matched = True
                    break

        # PHASE 2: default to field if no remote match
        if not matched:
            # Check all field rules including slope
            for _, rule in rules.iterrows():
                if rule['decision'] in ['strong field', 'weak field']:
                    matches_canopy = row['baseline_canopy'] == rule['baseline_canopy']
                    matches_target = row['target_sys'] == rule['target_sys']
                    matches_practice = row['practice'] == rule['practice']
                    matches_img_count = parse_condition(rule['img_count'], row['baseline_img_count'])
                    matches_slope = row['slope'] == rule['slope']

                    if (
                        matches_canopy
                        and matches_target
                        and matches_practice
                        and matches_img_count
                        and matches_slope
                    ):
                        decision = rule['decision']
                        matched = True
                        break

        if not matched:
            decision = 'review required'

        decisions.append(decision)

    df['decision'] = decisions
    
    desired_cols = [
    'project_id', 'poly_id', 'site_id', 'project_name',
    'plantstart', 'practice', 'target_sys', 'baseline_year',
    'ttc_2020', 'ttc_2021', 'ttc_2022', 'ttc_2023',  'ttc_2024', 
    'baseline_img_count', 'ev_img_count', 'baseline_canopy', 'ev_canopy', 
    'slope_area', 'slope', 'decision'
    ]
    df = df[desired_cols]

    # Summary
    summary = df['decision'].value_counts().reset_index()
    summary.columns = ['decision', 'count']
    summary['proportion'] = summary['count'] / len(df)
    print(summary)

    return df
