import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
import yaml

def apply_canopy_classification(params, df):
    """
    Assigns canopy classification for baseline and early verification
    based on the available ttc values and plantstart year.

    For each row:
    - Calculates the baseline year - calendar year prior to plant start and references 
    `ttc_{year}` for baseline statistics.
    - If plantstart is before 2020 or in the future, flags as 'invalid'.
    - If all TTC values are NaN, flags as 'NaN'.
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
    n_projects = df['project_id'].nunique()
    n_polys = df['poly_id'].nunique() 
    print(f"Analyzing canopy cover for {n_projects} projects and {n_polys} polygons...")

    criteria = params.get('criteria', {})
    canopy_thresh = criteria.get('canopy_threshold')

    current_year = datetime.today().year
    df['baseline_year'] = pd.to_datetime(df['plantstart'], errors='coerce').dt.year
    df['ev_year'] = pd.to_datetime(df['plantstart'], errors='coerce').dt.year + 2
    df.loc[:, 'baseline_canopy'] = 'unknown'
    df.loc[:, 'ev_canopy'] = 'unknown'

    ttc_cols = [col for col in df.columns if col.startswith('ttc_') and col[4:].isdigit()]

    # Tag plantstart years that are invalid (too early or in the future)
    invalid_years = (df['baseline_year'] < 2020) | (df['baseline_year'] > current_year)
    df.loc[invalid_years, ['baseline_canopy', 'ev_canopy']] = 'invalid'

    # Tag rows where all TTC values are NaN
    all_ttc_nan = df[ttc_cols].isna().all(axis=1)
    df.loc[all_ttc_nan, ['baseline_canopy', 'ev_canopy']] = np.nan

    # Eligible rows to process further
    eligible = ~(invalid_years | all_ttc_nan)

    for idx, row in df[eligible].iterrows():

        year = row['baseline_year']

        # Baseline classification
        baseline_year = year - 1
        baseline_col = f'ttc_{baseline_year}'
        if baseline_col in ttc_cols and pd.notna(row[baseline_col]):
            val = row[baseline_col]
            df.at[idx, 'baseline_canopy'] = 'closed' if val > canopy_thresh else 'open'
        else:
            df.at[idx, 'baseline_canopy'] = 'investigate'

        # EV classification
        plant_date = pd.to_datetime(row['plantstart'], errors='coerce')
        days_since_planting = (datetime.today() - plant_date).days

        if days_since_planting < 730:
            df.at[idx, 'ev_canopy'] = 'not available'
        else:
            ev_year = year + 2
            valid_ev_cols = [col for col in ttc_cols if int(col[4:]) >= ev_year and pd.notna(row[col])]
            if valid_ev_cols:
                closest_ev = min(valid_ev_cols, key=lambda col: int(col[4:]))
                val = row[closest_ev]
                df.at[idx, 'ev_canopy'] = 'closed' if val > canopy_thresh else 'open'
            else:
                df.at[idx, 'ev_canopy'] = 'investigate'
   
    return df