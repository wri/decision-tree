import numpy as np
import pandas as pd
from typing import Dict, List, Optional


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


# ---------------------------------------------------------------------------
# Project-level aggregation
# ---------------------------------------------------------------------------

def _score_one_mode(
    g: pd.DataFrame,
    score_col: str,
    dec_col: str,
    total_area: float,
    min_cov: float,
    mang_thr: float,
    rev_thr: float,
    missing_thr: float = 0.20,
    canopy_missing_col: str = "baseline_canopy_missing",
    img_missing_col: str = "baseline_img_missing",
) -> dict:
    """
    Compute score, label, coverage stats, and data quality flag for one mode
    (baseline or EV) for a single project group.

    Project label = majority DT decision by area among scoreable polygons.
    Suitability score = area-weighted average (supplemental signal only).

    Special label override order (fixed):
      1. mangrove        — overrides everything if ≥ mang_thr of area
      2. review required — overrides if ≥ rev_thr fraction of polygons are review required
      3. not available   — applied if fewer than min_cov fraction of polygons have a valid score

    Data quality flag
    -----------------
    After the label is determined, each missing-data component (canopy, imagery,
    slope) is checked against missing_thr. Flags are labelled HIGH or LOW based
    on whether the missing component is material to the assigned label:

      Field project  → slope missing = HIGH;  canopy / imagery missing = LOW
      Remote project → canopy + imagery missing = HIGH;  slope missing = LOW
      Other labels   → all missing components flagged without priority label
    """
    target_raw = g.get("target_sys", pd.Series(np.nan, index=g.index))
    dec_raw    = g.get(dec_col,      pd.Series(np.nan, index=g.index))

    target = target_raw.fillna("").astype(str).str.strip().str.lower()
    dec    = dec_raw.fillna("").astype(str).str.strip().str.lower()

    mang_mask = (target == "mangrove")
    rev_mask  = (dec == "review required")

    # Polygons with no decision are unscoreable — exclude from rollup
    missing_dec = dec_raw.isna()

    scored_mask      = g[score_col].notna() & ~missing_dec
    area_scored      = g.loc[scored_mask, "_area_clean_"].sum()
    coverage         = (area_scored / total_area) if total_area > 0 else 0.0
    frac_mang        = g.loc[mang_mask, "_area_clean_"].sum() / total_area if total_area > 0 else 0.0
    poly_frac_scored = round(scored_mask.sum() / len(g), 3) if len(g) > 0 else 0.0

    # Review required: use polygon count fraction so a single large polygon
    # cannot override the project label on its own.
    frac_rev = round(rev_mask.sum() / len(g), 3) if len(g) > 0 else 0.0

    raw_score = (
        (g.loc[scored_mask, score_col] * g.loc[scored_mask, "_area_clean_"]).sum() / area_scored
        if area_scored > 0 else np.nan
    )

    if poly_frac_scored < min_cov:
        score_pub, label = np.nan, "not available"
    else:
        score_pub = round(float(raw_score), 1) if pd.notna(raw_score) else np.nan
        label_area = g.loc[scored_mask, "_area_clean_"].groupby(dec[scored_mask]).sum()
        label = label_area.idxmax() if not label_area.empty else "not available"

    # Special label overrides — applied in fixed priority order
    if frac_mang >= mang_thr:
        label, score_pub = "mangrove", np.nan
    elif frac_rev >= rev_thr:
        label, score_pub = "review required", np.nan
    elif poly_frac_scored < min_cov:
        label, score_pub = "not available", np.nan

    # Convert remote suitability → label confidence.
    # For field-classified projects the raw score runs in the wrong direction
    # (low raw score = strong field = high confidence), so invert it so that
    # a high confidence value always means "continuous data strongly supports
    # the assigned label."
    if "field" in label.lower() and pd.notna(score_pub):
        score_pub = round(100.0 - score_pub, 1)

    # ------------------------------------------------------------------
    # Context-aware data quality flag
    # ------------------------------------------------------------------
    n_poly = len(g)

    def _missing_frac(col):
        if col in g.columns and n_poly > 0:
            return g[col].sum() / n_poly
        return 0.0

    frac_canopy_missing = _missing_frac(canopy_missing_col)
    frac_img_missing    = _missing_frac(img_missing_col)
    frac_slope_missing  = _missing_frac("slope_missing")

    label_lower = label.lower()
    is_field    = "field"  in label_lower
    is_remote   = "remote" in label_lower

    flags = []
    if is_field:
        # Slope is the component most material to a field classification
        if frac_slope_missing >= missing_thr:
            flags.append(
                f"slope missing for {frac_slope_missing:.0%} of polygons [HIGH: field project]"
            )
        if frac_canopy_missing >= missing_thr:
            flags.append(
                f"canopy missing for {frac_canopy_missing:.0%} of polygons [LOW: field project]"
            )
        if frac_img_missing >= missing_thr:
            flags.append(
                f"imagery missing for {frac_img_missing:.0%} of polygons [LOW: field project]"
            )
    elif is_remote:
        # Canopy and imagery are the components most material to a remote classification
        if frac_canopy_missing >= missing_thr:
            flags.append(
                f"canopy missing for {frac_canopy_missing:.0%} of polygons [HIGH: remote project]"
            )
        if frac_img_missing >= missing_thr:
            flags.append(
                f"imagery missing for {frac_img_missing:.0%} of polygons [HIGH: remote project]"
            )
        if frac_slope_missing >= missing_thr:
            flags.append(
                f"slope missing for {frac_slope_missing:.0%} of polygons [LOW: remote project]"
            )
    else:
        # mangrove / review required / not available — flag all missing generically
        if frac_canopy_missing >= missing_thr:
            flags.append(f"canopy missing for {frac_canopy_missing:.0%} of polygons")
        if frac_img_missing >= missing_thr:
            flags.append(f"imagery missing for {frac_img_missing:.0%} of polygons")
        if frac_slope_missing >= missing_thr:
            flags.append(f"slope missing for {frac_slope_missing:.0%} of polygons")

    data_quality_flag = "; ".join(flags) if flags else ""

    return {
        "conf":               score_pub,
        "label":              label,
        "pct_area_scored":    round(float(coverage), 3),
        "poly_frac_scored":   poly_frac_scored,
        "scored_area":        round(float(area_scored), 3),
        "frac_mangrove":      round(float(frac_mang), 3),
        "data_quality_flag":  data_quality_flag,
    }


def aggregate_project_score(
    params: Dict,
    df: pd.DataFrame,
    project_col: str = "project_id",
    area_col: Optional[str] = None,
    baseline_score_col: str = "baseline_remote_suitability",
    ev_score_col: str = "ev_remote_suitability",
    baseline_dec_col: str = "baseline_decision",
    ev_dec_col: str = "ev_decision",
    project_name_col: str = "project_name",
) -> pd.DataFrame:
    """
    Compute a single, area-weighted confidence score per project.

    Project label is derived from the majority DT decision by area across
    scoreable polygons. The suitability score is a supplemental signal only.

    Output columns
    --------------
    base_conf              : 0–100 label confidence score (supplemental).
                             For remote projects: higher = continuous data more
                             strongly supports remote. For field projects: score is
                             inverted (100 − raw) so higher still means more
                             confident in the field classification. NaN for mangrove,
                             review required, and not available.
    base_label             : project label from majority DT decision by area
    base_pct_area_scored   : fraction of project area with a valid polygon score
    base_poly_frac_scored  : fraction of polygons with a valid score
    base_scored_area       : total area (ha) of scored polygons
    base_frac_mangrove     : fraction of project area flagged as mangrove
    base_data_quality_flag : human-readable note when a component is missing for
                             ≥ missing_data_threshold of polygons; priority label
                             (HIGH/LOW) reflects materiality to the assigned label
    ev_*                   : same columns for the EV mode
    """
    cfg         = (params or {}).get("scoring", {})
    area_col    = area_col or cfg.get("area_col", "area")
    min_cov     = float(cfg.get("min_coverage",           0.60))
    mang_thr    = float(cfg.get("mangrove_frac_threshold", 0.10))
    rev_thr     = float(cfg.get("review_frac_threshold",   0.60))
    missing_thr = float(cfg.get("missing_data_threshold",  0.20))

    area = pd.to_numeric(df.get(area_col, np.nan), errors="coerce").clip(lower=0)

    missing_area = area.isna().sum()
    if missing_area > 0:
        print(f"[aggregate] WARNING: {missing_area} polygon(s) have no area value — "
              f"they will be excluded from area-weighted project scores.")

    work = df.copy()
    work["_area_clean_"] = area

    out_rows: List[Dict] = []
    for prj, g in work.groupby(project_col, dropna=False):
        total_area = g["_area_clean_"].sum()

        prj_name = (
            g[project_name_col].dropna().iloc[0]
            if project_name_col in g.columns and g[project_name_col].notna().any()
            else None
        )

        b = _score_one_mode(
            g, baseline_score_col, baseline_dec_col,
            total_area, min_cov, mang_thr, rev_thr,
            missing_thr=missing_thr,
            canopy_missing_col="baseline_canopy_missing",
            img_missing_col="baseline_img_missing",
        )
        e = _score_one_mode(
            g, ev_score_col, ev_dec_col,
            total_area, min_cov, mang_thr, rev_thr,
            missing_thr=missing_thr,
            canopy_missing_col="ev_canopy_missing",
            img_missing_col="ev_img_missing",
        )

        out_rows.append({
            project_col:      prj,
            project_name_col: prj_name,
            "total_area":     round(float(total_area), 3),

            "base_conf":               b["conf"],
            "base_label":              b["label"],
            "base_pct_area_scored":    b["pct_area_scored"],
            "base_poly_frac_scored":   b["poly_frac_scored"],
            "base_scored_area":        b["scored_area"],
            "base_frac_mangrove":      b["frac_mangrove"],
            "base_data_quality_flag":  b["data_quality_flag"],

            "ev_conf":               e["conf"],
            "ev_label":              e["label"],
            "ev_pct_area_scored":    e["pct_area_scored"],
            "ev_poly_frac_scored":   e["poly_frac_scored"],
            "ev_scored_area":        e["scored_area"],
            "ev_frac_mangrove":      e["frac_mangrove"],
            "ev_data_quality_flag":  e["data_quality_flag"],
        })

    return pd.DataFrame(out_rows)