from datetime import datetime

import numpy as np
import pandas as pd

def apply_canopy_classification(params, df):
    """
    Assigns canopy classification for baseline and early verification
    based on the available ttc values and plantstart year.

    For each row:
    - Calculates the baseline year - calendar year prior to plant start and references 
    `ttc_{year}` for baseline statistics.
    - If all TTC values are NaN, flags as 'NaN'.
    - If the TTC column for the expected baseline year is missing or has no data
      for that polygon, flags as 'investigate'.
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
    baseline_range = criteria.get("baseline_range")
    ev_range = criteria.get("ev_range")
    current_year = datetime.today().year

    df['plantstart_dt'] = pd.to_datetime(df['plantstart'], errors='coerce')
    df['baseline_year'] = df['plantstart_dt'].dt.year
    df.loc[:, 'baseline_canopy'] = 'unknown'
    df.loc[:, 'ev_canopy'] = 'unknown'

    #label missing ttc - TODO: this will move to an earlier step once the TTC package is finished
    ttc_cols = [col for col in df.columns if col.startswith('ttc_') and col[4:].isdigit() and len(col[4:]) == 4]
    null_mask = df[ttc_cols].isna().all(axis=1)  
    df.loc[null_mask, 'notes'] = 'missing-ttc' # this part moves, rest stays
    df.loc[null_mask, ['baseline_canopy', 'ev_canopy']] = np.nan

    # Eligible rows to process further
    eligible = ~(null_mask)

    for idx, row in df[eligible].iterrows():
        plant_date = row['plantstart_dt']

        # Baseline classification
        baseline_year = plant_date.year - 1
        baseline_col = f'ttc_{baseline_year}'

        if baseline_col in ttc_cols and pd.notna(row[baseline_col]):
            val = row[baseline_col]
            df.at[idx, 'baseline_canopy'] = 'closed' if val > canopy_thresh else 'open'
        else:
            df.at[idx, 'baseline_canopy'] = 'investigate'

        # EV classification
        days_since_planting = (datetime.today() - plant_date).days

        if days_since_planting < ev_range[0]:
            df.at[idx, 'ev_canopy'] = 'not available'
        else:  
            ev_year = plant_date.year + 2
            ev_col = f'ttc_{ev_year}'
            if ev_col in ttc_cols and pd.notna(row[ev_col]):
                val = row[ev_col]
                df.at[idx, 'ev_canopy'] = 'closed' if val > canopy_thresh else 'open'
            else:
                df.at[idx, 'ev_canopy'] = 'investigate'
    return df