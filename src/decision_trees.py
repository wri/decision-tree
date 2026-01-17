import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import os

from src.constants import DESIRED_COLS


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
      • assign first_decision ∈ {mangrove, remote, field}  
        (intentionally ignores slope entirely)
    
    PHASE 2:
      • for first_decision=='remote', look only at img_count rules  
      • for first_decision=='field', look only at slope rules  
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
    base_rules = rules[['canopy','target_sys','practice','img_count', 'first_decision']]
    remote_rules = rules[rules['first_decision']=='remote'][[
        'canopy','target_sys','practice','img_count','final_decision'
    ]]
    field_rules  = rules[rules['first_decision']=='field'][[
        'canopy','target_sys','practice','slope','final_decision'
    ]]

    decisions = []
    for _, row in df.iterrows():

        # PHASE 1: first_decision — assign 'mangrove', 'remote' or 'field'
        if row['target_sys'] == 'mangrove':
            base = 'mangrove'
        else:
            base = None
            for _, rule in base_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys']      == rule['target_sys'] and
                    row['practice']        == rule['practice'] and
                    parse_condition(rule['img_count'], row['baseline_img_count'])
                ):
                    base = rule['first_decision']
                    break

        # ——— PHASE 2: final_decision — refine using img_count or slope only
        if base == 'remote':
            # only img_count drives strong vs. weak remote
            final = None
            for _, rule in remote_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
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
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    row['slope'] == rule['slope']
                ):
                    final = rule['final_decision']
                    break

        else:
            final = base

        decisions.append(final or 'review required')

    df['baseline_decision'] = decisions

    df = df.filter(items=DESIRED_COLS)

    # Summary
    summary = df['baseline_decision'].value_counts().reset_index()
    summary.columns = ['baseline_decision', 'count']
    summary['proportion'] = round((summary['count'] / len(df))*100)
    print("\nBaseline Decision Summary:\n")
    print(summary)

    return df

def apply_rules_ev(params, df):
    """
    Decision tree for early verification (EV) classification.

    PHASE 1:
      • assign first_decision ∈ {mangrove, remote, field}
    
    PHASE 2:
      • for first_decision=='remote', look only at img_count rules  
      • for first_decision=='field', look only at slope rules  
      • mangrove stays as is (final_decision = 'mangrove')

    Additional Rule:
      • if EV year is current or future, decision is 'not available'
    """
    df = df.copy()
    rules = pd.read_csv(params['criteria']['rules'])

    # Clean rules df
    for col in rules.columns:
        if rules[col].dtype == object:
            rules[col] = rules[col].map(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )

    # Clean input df
    for col in ['ev_canopy', 'target_sys', 'practice', 'slope']:
        df.loc[:, col] = df[col].astype(str).str.strip().str.lower()


    # Filter EV-specific rule columns
    base_rules = rules[['canopy','target_sys','practice','img_count', 'first_decision']]
    remote_rules = rules[rules['first_decision']=='remote'][[
        'canopy','target_sys','practice','img_count','final_decision'
    ]]
    field_rules = rules[rules['first_decision']=='field'][[
        'canopy','target_sys','practice','slope','final_decision'
    ]]

    today = datetime.today()
    ev_days_start, ev_days_end = params['criteria']['ev_range']

    decisions = []

    for _, row in df.iterrows():
        if pd.isna(row['plantstart']):
            decisions.append('not available')
            continue

        plant_start = pd.to_datetime(row['plantstart'])
        ev_window_start = plant_start + timedelta(days=ev_days_start)
        if today < ev_window_start:
            decisions.append('not available')
            continue

        # PHASE 1 — first_decision: 'mangrove', 'remote', or 'field'
        # NOTE: EV will use the baseline canopy if enough time has passed
        # that we need to verify
        if row['target_sys'] == 'mangrove':
            base = 'mangrove'
        else:
            base = None
            for _, rule in base_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    parse_condition(rule['img_count'], row['ev_img_count'])
                ):
                    base = rule['first_decision']
                    break

        # PHASE 2 — final_decision
        if base == 'remote':
            final = None
            for _, rule in remote_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    parse_condition(rule['img_count'], row['ev_img_count'])
                ):
                    final = rule['final_decision']
                    break

        elif base == 'field':
            final = None
            for _, rule in field_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    row['slope'] == rule['slope']
                ):
                    final = rule['final_decision']
                    break
        else:
            final = base

        decisions.append(final or 'review required')

    # Optional: include only necessary columns (mirror structure)
    df = df.filter(items=DESIRED_COLS)

    # Add column for ev decision
    df['ev_decision'] = decisions

    # Summary
    summary = df['ev_decision'].value_counts().reset_index()
    summary.columns = ['ev_decision', 'count']
    summary['proportion'] = round((summary['count'] / len(df)) * 100)
    print("\nEV Decision Summary:\n")
    print(summary)

    return df