from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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
    slope) is checked against missing_thr. Flags are labeled HIGH or LOW based
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