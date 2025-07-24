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

def apply_rules_baseline(params, df):
    """
    Decision tree for baseline classification.
    
    PHASE 1:
      • assign base_decision ∈ {mangrove, remote, field}  
        (intentionally ignores slope entirely)
    
    PHASE 2:
      • for base_decision=='remote', look only at img_count rules  
      • for base_decision=='field', look only at slope rules  
      • mangrove stays as is (final_decision = 'mangrove')
    """

    rules = pd.read_csv(params['criteria']['rules'])

    # clean rules df
    for col in rules.columns:
        if rules[col].dtype == object:
            rules[col] = rules[col].map(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )

    # clean input df
    for col in ['baseline_canopy', 'target_sys', 'practice', 'slope']:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # 4) split the rules into exactly the subsets each phase needs
    base_rules = rules[['baseline_canopy','target_sys','practice','img_count','base_decision']]
    remote_rules = rules[rules['base_decision']=='remote'][[
        'baseline_canopy','target_sys','practice','img_count','final_decision'
    ]]
    field_rules  = rules[rules['base_decision']=='field'][[
        'baseline_canopy','target_sys','practice','slope','final_decision'
    ]]

    decisions = []
    for _, row in df.iterrows():

        # PHASE 1: base_decision — assign 'mangrove', 'remote' or 'field'
        if row['target_sys'] == 'mangrove':
            base = 'mangrove'
        else:
            base = None
            for _, rule in base_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['baseline_canopy'] and
                    row['target_sys']       == rule['target_sys'] and
                    row['practice']         == rule['practice'] and
                    parse_condition(rule['img_count'], row['baseline_img_count'])
                ):
                    base = rule['base_decision']
                    break

        # ——— PHASE 2: final_decision — refine using img_count or slope only
        if base == 'remote':
            # only img_count drives strong vs. weak remote
            final = None
            for _, rule in remote_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['baseline_canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    parse_condition(rule['img_count'], row['baseline_img_count'])
                ):
                    final = rule['final_decision']
                    break

        elif base == 'field':
            # only slope drives strong vs. weak field
            final = None
            for _, rule in field_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['baseline_canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    row['slope'] == rule['slope']
                ):
                    final = rule['final_decision']
                    break

        else:
            final = base

        decisions.append(final or 'review required')

    df['decision'] = decisions

    desired_cols = [
    'project_id', 'poly_id', 'site_id', 'project_name',
    'plantstart', 'practice', 'target_sys', 'baseline_year',
    'ttc_2020', 'ttc_2021', 'ttc_2022', 'ttc_2023',  'ttc_2024', 
    'baseline_img_count', 'ev_img_count', 'baseline_canopy', 
    'ev_canopy', 'slope_area', 'slope', 'decision'
    ]
    df = df[desired_cols]

    # Summary
    summary = df['decision'].value_counts().reset_index()
    summary.columns = ['decision', 'count']
    summary['proportion'] = round((summary['count'] / len(df))*100)
    print(summary)

    return df
