import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
import yaml

def apply_canopy_classification(df,
                                param_path):
    """
    Assigns canopy classification for baseline and early verification
    based on the available ttc values and plantstart year.

    For each row:
    - Looks for the nearest valid `ttc_{year}` column to the plantstart year.
    - If plantstart is outside acceptable bounds (before 2020 or in the future), flags as 'invalid'.
    - If all TTC values are NaN, flags as 'invalid'.
    - If TTC data exists but not in the expected time window, flags as 'investigate'.
    - If planting occurred too recently for EV, sets 'ev_canopy' as 'not available'.
    - Otherwise, classifies as 'open' or 'closed' using the `canopy_thresh`.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with ttc columns and a plantstart column.
    - canopy_thresh (float): Threshold for distinguishing open vs closed canopy.
    - baseline_range (tuple): Days relative to plantstart defining the baseline period.
    - ev_range (tuple): Days relative to plantstart defining the early verification period.

    Returns:
    - pd.DataFrame: Original dataframe with two new columns: `baseline_canopy` and `ev_canopy`.
    """
    with open(param_path) as file:
        params = yaml.safe_load(file)
    
    criteria = params.get('criteria', {})
    canopy_thresh = criteria.get('canopy_threshold')

    current_year = datetime.today().year
    df['plantstart_year'] = pd.to_datetime(df['plantstart'], errors='coerce').dt.year
    df.loc[:, 'baseline_canopy'] = 'unknown'
    df.loc[:, 'ev_canopy'] = 'unknown'

    ttc_cols = [col for col in df.columns if col.startswith('ttc_') and col[4:].isdigit()]

    # Tag plantstart years that are invalid (too early or in the future)
    invalid_years = (df['plantstart_year'] < 2020) | (df['plantstart_year'] > current_year)
    df.loc[invalid_years, ['baseline_canopy', 'ev_canopy']] = 'invalid'

    # Tag rows where all TTC values are NaN
    all_ttc_nan = df[ttc_cols].isna().all(axis=1)
    df.loc[all_ttc_nan, ['baseline_canopy', 'ev_canopy']] = 'invalid'

    # Eligible rows to process further
    eligible = ~(invalid_years | all_ttc_nan)

    for idx, row in df[eligible].iterrows():
        year = row['plantstart_year']

        # Baseline classification
        valid_baseline_cols = [col for col in ttc_cols if int(col[4:]) <= year and pd.notna(row[col])]
        if valid_baseline_cols:
            closest_baseline = max(valid_baseline_cols, key=lambda col: int(col[4:]))
            val = row[closest_baseline]
            df.at[idx, 'baseline_canopy'] = 'closed' if val > canopy_thresh else 'open'
        else:
            df.at[idx, 'baseline_canopy'] = 'investigate'

        # EV classification
        ev_target_year = year + 2
        years_since_planting = datetime.today().year - year
        
        # not available if less than 2 years since planting
        if years_since_planting < 2:
            df.at[idx, 'ev_canopy'] = 'not available'
        else:
            valid_ev_cols = [col for col in ttc_cols if int(col[4:]) >= ev_target_year and pd.notna(row[col])]
            if valid_ev_cols:
                closest_ev = min(valid_ev_cols, key=lambda col: int(col[4:]))
                val = row[closest_ev]
                df.at[idx, 'ev_canopy'] = 'closed' if val > canopy_thresh else 'open'
            else:
                df.at[idx, 'ev_canopy'] = 'investigate'

    return df