import pandas as pd
import numpy as np

def calc_cost_to_verify(params, df, area_col="area", decimals=1):
    '''
    Using the remote vs field verification assignment and the polygon area,
    calculate the baseline and EV verification costs for the project.

    Rules:
      - For each phase, use that phase's decision (remote/field) to pick the rate.
      - If the decision is "mangrove" or "not available", cost is $0.
      - Costs are rounded to `decimals` (default 1 = nearest tenth).
      - If `save_path` is provided, the updated df is saved to CSV.
    '''
    r_price = params['cost']['field']
    f_price = params['cost']['field']
    # normalize decisions and area
    base_dec = df["baseline_decision"].astype(str).str.strip().str.lower()
    ev_dec   = df["ev_decision"].astype(str).str.strip().str.lower()
    area     = pd.to_numeric(df[area_col], errors="coerce").fillna(0.0)

    remote_set = {"weak remote", "strong remote"}
    field_set  = {"weak field", "strong field"}
    zero_set   = {"mangrove", "not available"}

    # initialize columns
    df["baseline_cost"] = 0.0
    df["ev_cost"] = 0.0

    # Baseline cost: use baseline_decision's method
    df.loc[base_dec.isin(remote_set), "baseline_cost"] = area[base_dec.isin(remote_set)] * r_price
    df.loc[base_dec.isin(field_set),  "baseline_cost"] = area[base_dec.isin(field_set)]  * f_price
    df.loc[base_dec.isin(zero_set),   "baseline_cost"] = 0.0

    # EV cost: use ev_decision's method
    df.loc[ev_dec.isin(remote_set), "ev_cost"] = area[ev_dec.isin(remote_set)] * r_price
    df.loc[ev_dec.isin(field_set),  "ev_cost"] = area[ev_dec.isin(field_set)]  * f_price
    df.loc[ev_dec.isin(zero_set),   "ev_cost"] = 0.0

    # Round to nearest tenth (or whatever `decimals` you pass)
    if decimals is not None:
        df["baseline_cost"] = df["baseline_cost"].round(decimals)
        df["ev_cost"] = df["ev_cost"].round(decimals)

    # Totals by phase
    totals = {
        "baseline_total": float(df["baseline_cost"].sum()),
        "ev_total": float(df["ev_cost"].sum()),
        "grand_total": float(df["baseline_cost"].sum() + df["ev_cost"].sum()),
    }

    # Nicely formatted portfolio totals
    print(
        "\nPortfolio Totals\n"
        "-----------------\n"
        f"{'Baseline total':<16} ${totals['baseline_total']:,.2f}\n"
        f"{'EV total':<16} ${totals['ev_total']:,.2f}\n"
        f"{'Grand total':<16} ${totals['grand_total']:,.2f}\n"
    )
    return df