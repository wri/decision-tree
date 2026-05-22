import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re

def parse_condition(value, actual):
    """
    Evaluate simple condition like '>=1', '<2', or direct value.
    Interprets and evaluates conditional expressions (like ">=1") 
    from the rules CSV.
    """
    if isinstance(value, str) and any(op in value for op in [">=", "<=", ">", "<", "=="]):
        value = value.strip()
        if not re.fullmatch(r"(>=|<=|>|<|==)-?\d+(\.\d+)?", value):
            raise ValueError(f"Malformed condition string: {value!r}")
        try:
            return eval(f"{actual} {value}")
        except Exception:
            return False
    return actual == value

def _image_timing_tier(row):
    """
    Classify a polygon's image availability into a timing tier.

    Returns:
    - 'hq_1yr'   : at least one image within 1-year baseline window
    - 'ext_18mo' : no image within 1 year, but at least one within 1.5 years
    - 'none'     : no usable imagery in either window
    """
    if row['baseline_img_count'] >= 1:
        return 'hq_1yr'
    elif row['baseline_ext_img_count'] >= 1:
        return 'ext_18mo'
    else:
        return 'none'

def apply_rules_baseline(rules_file_path, df):
    """
    Decision tree for baseline classification.

    * If a polygon was already flagged as problematic during cleaning, 
    carry that flag through as its decision by ref 'notes' column
    
    PHASE 1:
      • assign first_decision {mangrove, remote, field}  
        (intentionally ignores slope entirely)
    
    PHASE 2:
      • for first_decision=='remote', final_decision is driven by both
        img_count rules AND the timing tier of available imagery:
          - 'hq_1yr'   : image exists within 1yr  -> apply rules normally
          - 'ext_18mo' : image only within 1.5yrs  -> cap at 'weak remote'
          - 'none'     : no imagery in either window -> 'review required'
      • for first_decision=='field', look only at slope rules  
      • mangrove stays as is (final_decision = 'mangrove')
    """
    df = df.copy()
    rules = pd.read_csv(rules_file_path)

    # Clean rules df
    for col in rules.columns:
        if rules[col].dtype == object:
            rules[col] = rules[col].map(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )

    # Clean input df
    for col in ['baseline_canopy', 'target_sys', 'practice', 'slope']:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Pre-compute timing tier for each polygon
    df['img_tier'] = df.apply(_image_timing_tier, axis=1)

    # Split rules into subsets each phase needs
    base_rules = rules[['canopy', 'target_sys', 'practice', 'img_count', 'first_decision']]
    remote_rules = rules[rules['first_decision'] == 'remote'][[
        'canopy', 'target_sys', 'practice', 'img_count', 'final_decision'
    ]]
    field_rules = rules[rules['first_decision'] == 'field'][[
        'canopy', 'target_sys', 'practice', 'slope', 'final_decision'
    ]]

    decisions = []
    for _, row in df.iterrows():

        # check notes column before assigning decision
        if pd.notna(row['notes']):
            decisions.append(row['notes'])
            continue

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
                    parse_condition(rule['img_count'], row['baseline_ext_img_count'])
                ):
                    base = rule['first_decision']
                    break

        # ——— PHASE 2: final_decision — refine using img_count, timing and slope
        if base == 'remote':
            tier = row['img_tier']

            if tier == 'hq_1yr':
                # Ideal case: image within 1 year — apply rules normally
                final = None
                for _, rule in remote_rules.iterrows():
                    if (
                        row['baseline_canopy'] == rule['canopy'] and
                        row['target_sys']      == rule['target_sys'] and
                        row['practice']        == rule['practice'] and
                        parse_condition(rule['img_count'], row['baseline_img_count'])
                    ):
                        final = rule['final_decision']
                        break
                final = final or 'review required'

            elif tier == 'ext_18mo':
                # Flexible: image exists within 1.5 years but not within 1 year
                # Cap at weak remote regardless of count
                final = 'weak remote'

            else:
                # No imagery in either window — remote sensing not possible,
                # fall back to field visit using slope rules
                final = None
                for _, rule in field_rules.iterrows():
                    if (
                        row['baseline_canopy'] == rule['canopy'] and
                        row['target_sys']      == rule['target_sys'] and
                        row['practice']        == rule['practice'] and
                        row['slope']           == rule['slope']
                    ):
                        final = rule['final_decision']
                        break
                final = final or 'review required'

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

    # Summary
    summary = df['baseline_decision'].value_counts().reset_index()
    summary.columns = ['baseline_decision', 'count']
    summary['proportion'] = round((summary['count'] / len(df))*100, 2)
    print("\nBaseline Decision Summary (polygon scale):\n")
    print(summary)

    return df

def apply_rules_ev(params, rules_file_path, df):
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
    rules = pd.read_csv(rules_file_path)

    # Clean rules df
    for col in rules.columns:
        if rules[col].dtype == object:
            rules[col] = rules[col].map(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )

    # Clean input df
    for col in ['target_sys', 'practice', 'slope']:
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

    # Add column for ev decision
    df['ev_decision'] = decisions

    # Summary
    summary = df['ev_decision'].value_counts().reset_index()
    summary.columns = ['ev_decision', 'count']
    summary['proportion'] = round((summary['count'] / len(df)) * 100)
    # print("\nEV Decision Summary:\n")
    # print(summary)

    return df


# ---------------------------------------------------------------------------
# Polygon-level scoring
# ---------------------------------------------------------------------------

def _get_ttc_for_year(df: pd.DataFrame, year_series: pd.Series) -> pd.Series:
    """
    Extract the per-row TTC value using a Series of integer years.

    Looks for columns named ttc_{YYYY} and returns the value from the column
    whose year matches year_series for each row. Returns NaN where no match.
    """
    ttc_year_cols = {
        int(col[4:]): col
        for col in df.columns
        if col.startswith("ttc_") and col[4:].isdigit() and len(col[4:]) == 4
    }
    result = pd.Series(np.nan, index=df.index)
    for year, col in ttc_year_cols.items():
        mask = (year_series == year) & df[col].notna()
        result[mask] = df.loc[mask, col]
    return result


def apply_scoring(
    params: dict,
    df: pd.DataFrame,
    baseline_col: str = "baseline_decision",
    ev_col: str = "ev_decision",
) -> pd.DataFrame:
    """
    Compute a Remote Suitability score (0–100) per polygon for baseline and EV.

    The score is a supplemental gradient signal only — the decision tree
    (baseline_decision / ev_decision) remains the authoritative classification.
    It answers: given this polygon's continuous feature values, how strongly
    do they lean toward remote vs. field verification?

        Higher score → stronger remote candidate
        Lower score  → stronger field candidate

    The score is driven by three continuous variables only:

      1. Canopy cover (TTC)   — lower TTC = more open = stronger remote
      2. Image count + timing — more images, closer to planting = stronger remote;
                                 images in the extended 1.5yr window score at half
                                 the weight of images in the core 1yr window
      3. Slope                — higher % steep area = weaker field classification

    All weights and normalisation caps are configurable in params under 'scoring'.

    Polygons classified as mangrove, review required, or not available receive
    NaN scores and are excluded from project-level rollups. Mask flags are
    computed internally and are NOT written to the output DataFrame.

    Outputs added to df
    -------------------
    baseline_remote_suitability     : 0–100 score (baseline)
    baseline_remote_suitability_wtd : score × area (baseline)
    ev_remote_suitability           : 0–100 score (EV)
    ev_remote_suitability_wtd       : score × area (EV)
    """
    cfg = (params or {}).get("scoring", {})

    area_col       = cfg.get("area_col", "area")
    w_canopy       = float(cfg.get("weight_canopy",   0.40))
    w_img          = float(cfg.get("weight_img",      0.40))
    w_slope        = float(cfg.get("weight_slope",    0.20))
    max_imgs       = float(cfg.get("max_imgs",        5.0))
    img_1yr_wt     = float(cfg.get("img_1yr_weight",  1.0))
    img_ext_wt     = float(cfg.get("img_ext_weight",  0.5))

    # ------------------------------------------------------------------
    # Canopy score  (0 = fully closed → score 0;  100 = fully open → score 100)
    # baseline: ttc at plantstart_year - 1
    # ev:       ttc at plantstart_year + 2
    # ------------------------------------------------------------------
    base_years = pd.to_numeric(df.get("baseline_year", np.nan), errors="coerce")

    baseline_ttc = _get_ttc_for_year(df, base_years - 1)
    ev_ttc       = _get_ttc_for_year(df, base_years + 2)

    canopy_base = (100.0 - pd.to_numeric(baseline_ttc, errors="coerce")).clip(0, 100)
    canopy_ev   = (100.0 - pd.to_numeric(ev_ttc,       errors="coerce")).clip(0, 100)

    # ------------------------------------------------------------------
    # Image score  (0 = no imagery → score 0;  max_imgs in 1yr → score 100)
    # Extension-only images (between 1yr and 1.5yr) score at img_ext_weight.
    # Missing or unresolvable image counts remain NaN and propagate to NaN score.
    # ------------------------------------------------------------------
    _nan_col = pd.Series(np.nan, index=df.index)

    for _col in ["baseline_img_count", "baseline_ext_img_count", "ev_img_count"]:
        if _col not in df.columns:
            print(f"[scoring] WARNING: column '{_col}' not found — "
                  f"image score will be NaN for all polygons.")

    img_1yr      = pd.to_numeric(df.get("baseline_img_count",     _nan_col), errors="coerce").clip(lower=0)
    img_ext_tot  = pd.to_numeric(df.get("baseline_ext_img_count", _nan_col), errors="coerce").clip(lower=0)
    img_ext_only = (img_ext_tot - img_1yr).clip(lower=0)
    img_ev       = pd.to_numeric(df.get("ev_img_count",           _nan_col), errors="coerce").clip(lower=0)

    img_base = (
        img_1yr_wt * (img_1yr      / max_imgs).clip(0, 1) +
        img_ext_wt * (img_ext_only / max_imgs).clip(0, 1)
    ) / (img_1yr_wt + img_ext_wt) * 100.0

    img_ev_score = (img_ev / max_imgs).clip(0, 1) * 100.0

    # ------------------------------------------------------------------
    # Slope score  (slope_area = % of polygon area that is steep, 0–100)
    # Higher steep fraction = field classification less strong = higher score.
    # Missing slope data remains NaN and propagates to NaN score.
    # ------------------------------------------------------------------
    if "slope_area" not in df.columns:
        print("[scoring] WARNING: column 'slope_area' not found — "
              "slope score will be NaN for all polygons.")

    slope_score = pd.to_numeric(df.get("slope_area", _nan_col), errors="coerce").clip(0, 100)

    # ------------------------------------------------------------------
    # Per-polygon missing component flags
    # Written to df so they are available in poly_output and can be
    # aggregated into project-level data quality flags downstream.
    # ------------------------------------------------------------------
    df["baseline_canopy_missing"] = baseline_ttc.isna()
    df["ev_canopy_missing"]       = ev_ttc.isna()
    df["baseline_img_missing"]    = img_1yr.isna() | img_ext_tot.isna()
    df["ev_img_missing"]          = img_ev.isna()
    df["slope_missing"]           = slope_score.isna()

    # ------------------------------------------------------------------
    # Weighted average — reweighted dynamically per polygon
    #
    # If a component is NaN for a given polygon (e.g. slope data unavailable),
    # that component's weight is excluded from the denominator so the score
    # is still computed from the remaining valid components. A polygon only
    # receives NaN if every component is NaN.
    # ------------------------------------------------------------------
    def _weighted_mean(components: dict, weight_map: dict) -> pd.Series:
        """
        Compute a per-row weighted average over a dict of named component Series.
        NaN components are excluded per row; weight denominator adjusts accordingly.
        """
        comp_df   = pd.DataFrame(components)
        weight_s  = pd.Series(weight_map)
        numer     = (comp_df * weight_s).sum(axis=1, skipna=True)
        denom     = comp_df.notna().multiply(weight_s).sum(axis=1)
        return numer / denom   # NaN only when denom == 0 (all components NaN)

    score_base = _weighted_mean(
        {"canopy": canopy_base, "img": img_base,      "slope": slope_score},
        {"canopy": w_canopy,    "img": w_img,          "slope": w_slope},
    )
    score_ev = _weighted_mean(
        {"canopy": canopy_ev,   "img": img_ev_score,  "slope": slope_score},
        {"canopy": w_canopy,    "img": w_img,          "slope": w_slope},
    )

    # ------------------------------------------------------------------
    # Null mask — computed locally, NOT written to output DataFrame
    # Polygons with missing decision or target_sys values are treated as
    # unscoreable (NaN) rather than receiving a misleading score.
    # ------------------------------------------------------------------
    for _col in ["target_sys", baseline_col, ev_col]:
        if _col not in df.columns:
            print(f"[scoring] WARNING: column '{_col}' not found — "
                  f"all polygons will be treated as unscoreable.")

    target_raw   = df.get("target_sys", pd.Series(np.nan, index=df.index))
    base_dec_raw = df.get(baseline_col, pd.Series(np.nan, index=df.index))
    ev_dec_raw   = df.get(ev_col,       pd.Series(np.nan, index=df.index))

    target   = target_raw.fillna("").astype(str).str.strip().str.lower()
    base_dec = base_dec_raw.fillna("").astype(str).str.strip().str.lower()
    ev_dec   = ev_dec_raw.fillna("").astype(str).str.strip().str.lower()

    base_null = (
        (target == "mangrove") |
        (base_dec == "review required") |
        (base_dec == "not available") |
        base_dec_raw.isna()
    )
    ev_null = (
        (target == "mangrove") |
        (ev_dec == "review required") |
        (ev_dec == "not available") |
        ev_dec_raw.isna()
    )

    score_base = score_base.round(1)
    score_ev   = score_ev.round(1)
    score_base[base_null] = np.nan
    score_ev[ev_null]     = np.nan

    # ------------------------------------------------------------------
    # Area-weighted variant for project rollups
    # ------------------------------------------------------------------
    area = pd.to_numeric(df.get(area_col, np.nan), errors="coerce").clip(lower=0)

    missing_area = area.isna().sum()
    if missing_area > 0:
        print(f"[scoring] WARNING: {missing_area} polygon(s) have no area value — "
              f"area-weighted scores will be NaN for those rows.")

    df["baseline_remote_suitability"]     = score_base
    df["baseline_remote_suitability_wtd"] = (score_base * area).round(1)
    df["ev_remote_suitability"]           = score_ev
    df["ev_remote_suitability_wtd"]       = (score_ev * area).round(1)

    return df