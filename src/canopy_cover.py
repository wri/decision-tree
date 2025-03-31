import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime


def apply_canopy_classification(df,
                                canopy_thresh,
                                baseline_range=(-365, 0),
                                ev_range=(730, 1095)):
    """
    Assigns canopy classification for baseline and early verification
    based on the available ttc values and plantstart year.

    For each row:
    - Looks for the nearest valid `ttc_{year}` column to the plantstart year.
    - If plantstart is outside of acceptable bounds (before 2020 or in the future), flags as 'investigate'.
    - If no valid TTC data exists, flags as 'unknown'.
    - If TTC data exists but not in the expected time window, flags as 'investigate'.
    - Otherwise, classifies as 'open' or 'closed' using the `canopy_thresh`.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with ttc columns and a plantstart column.
    - canopy_thresh (float): Threshold for distinguishing open vs closed canopy.
    - baseline_range (tuple): Days relative to plantstart defining the baseline period.
    - ev_range (tuple): Days relative to plantstart defining the early verification period.

    Returns:
    - pd.DataFrame: Original dataframe with two new columns: `baseline_canopy` and `ev_canopy`.
    """
    current_year = datetime.today().year
    df['plantstart_year'] = pd.to_datetime(df['plantstart'], errors='coerce').dt.year
    df.loc[:, 'baseline_canopy'] = 'unknown'
    df.loc[:, 'ev_canopy'] = 'unknown'

    # Identify invalid years
    invalid_years = (df['plantstart_year'] < 2020) | (df['plantstart_year'] > current_year)
    df.loc[invalid_years, ['baseline_canopy', 'ev_canopy']] = 'investigate'

    # Define valid rows to proceed with
    eligible = (df['plantstart_year'] >= 2020) & (df['plantstart_year'] <= current_year)

    # Proceed with classification for valid years
    for idx, row in df[eligible].iterrows():
        year = row['plantstart_year']
        ttc_cols = [col for col in df.columns if col.startswith('ttc_') and col[4:].isdigit()]

        # Baseline classification
        # gets a list of baseline options where the row doesnt have NaN
        valid_baseline_cols = [col for col in ttc_cols if int(col[4:]) <= year and pd.notna(row[col])]
        if valid_baseline_cols:
            # chooses the item with the highest numeric year
            closest_baseline = max(valid_baseline_cols, key=lambda col: int(col[4:]))
            val = row[closest_baseline]
            df.at[idx, 'baseline_canopy'] = 'closed' if val > canopy_thresh else 'open'
        else:
            df.at[idx, 'baseline_canopy'] = 'investigate'

        # Early verification classification
        ev_target_year = year + 2 
        valid_ev_cols = [col for col in ttc_cols if int(col[4:]) >= ev_target_year and pd.notna(row[col])]
        if valid_ev_cols:
            closest_ev = min(valid_ev_cols, key=lambda col: int(col[4:]))
            val = row[closest_ev]
            df.at[idx, 'ev_canopy'] = 'closed' if val > canopy_thresh else 'open'
        else:
            df.at[idx, 'ev_canopy'] = 'investigate'

    return df