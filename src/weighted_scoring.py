import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional

_OP_RE = re.compile(r'^\s*(>=|<=|==|>|<)\s*([-+]?\d+(?:\.\d+)?)\s*$')

def _cond_mask(series: pd.Series, cond) -> pd.Series:
    """
    Build a boolean mask for rows of `series` that satisfy `cond`.

    `cond` can be:
      - a comparison string like ">=2", "<1", "==3" (numeric compare)
      - an exact literal like "open", "closed" (case/space-normalized)

    Non-numeric cells become NaN for numeric compares and evaluate False.
    """
    m = _OP_RE.match(str(cond))
    if m:
        op, val = m.group(1), float(m.group(2))
        s = pd.to_numeric(series, errors="coerce")
        return {">=": s >= val, "<=": s <= val, "==": s == val, ">": s > val, "<": s < val}[op]
    return series.astype(str).str.strip().str.lower() == str(cond).strip().lower()


def _map_z_to_0_100(z: np.ndarray, scale_cfg: dict) -> np.ndarray:
    """
    Map raw axis score z -> [0,100].

    scale_cfg:
      method: "logistic" or "minmax"
      k: slope for logistic (default 0.8)
      min, max: expected min/max of z for minmax scaling
    """
    method = (scale_cfg or {}).get("method", "logistic")
    if method == "minmax":
        mn = float((scale_cfg or {}).get("min", -6))
        mx = float((scale_cfg or {}).get("max", 6))
        s = (z - mn) / (mx - mn)
        return np.clip(s, 0, 1) * 100.0
    # logistic by default: squashes to (0,1)
    k = float((scale_cfg or {}).get("k", 0.8))
    s = 1.0 / (1.0 + np.exp(-k * z))
    return s * 100.0


def apply_scoring(
    params: dict,
    df: pd.DataFrame,
    baseline_col: str = "baseline_decision",
    ev_col: str = "ev_decision",
) -> pd.DataFrame:
    """
    Compute a single “Remote Suitability” score per polygon for baseline and EV.

    Concept
    -------
    - We build a raw axis score z by summing feature contributions:
        z > 0  => more suitable for REMOTE verification
        z < 0  => more suitable for FIELD verification
    - Then we map z to [0,100] using logistic (default) or minmax scaling.
    - We also produce an area-weighted variant for project roll-ups:
        suitability_weighted = suitability_0_100 * area

    Inputs
    ------
    df : DataFrame with your existing columns (canopy, slope, *_img_count, etc.)
    params['suitability'] :
      - area_col: name of area column (default 'area')
      - scale: mapping method config (see _map_z_to_0_100)
      - base_bonus: optional nudges from your discrete decisions
      - features: dict with groups 'common', 'baseline_only', 'ev_only'
                  Each feature maps values/conditions -> scalar delta added to z.
                  Positive delta pushes toward remote; negative toward field.
                  Use `null` for values you want to ignore here (no delta).

    Outputs (added columns)
    -----------------------
    baseline_raw               : raw axis score (baseline)
    baseline_score             : z mapped to [0,100] (baseline)
    baseline_score_wtd         : area-weighted score (baseline)
    baseline_simple_bin        : simple label from thresholds (remote/field)

    ev_raw, ev_score, ev_score_wtd, ev_simple_bin

    Special handling
    ----------------
    - If target_sys == 'mangrove' => scores set to NaN (not part of R/F axis).
    - If a row is 'not available' for EV, we leave the EV score as NaN to avoid
      implying a field bias when the window/images simply aren’t ready.

    Thresholds
    ----------
    The simple label uses these defaults on 0..100 (tune if desired):
      >= 70  => "strong remote"
      55-70  => "weak remote"
      30-55  => "weak field"
      <  30  => "strong field"
    """
    cfg = (params or {}).get("suitability", {})
    area_col = cfg.get("area_col", "area")
    scale_cfg = cfg.get("scale", {})
    base_bonus = cfg.get("base_bonus", {}) or {}
    feats = cfg.get("features", {}) or {}
    common = feats.get("common", {}) or {}
    base_only = feats.get("baseline_only", {}) or {}
    ev_only = feats.get("ev_only", {}) or {}

    # Helpers to build z for a given mode (baseline/ev)
    def _z_for(mode_col: str, extra_feats: dict, prefix: str):
        z = np.zeros(len(df), dtype=float)

        # 1) Add contributions from configured features (vectorized by masks)
        for feat, value_map in {**common, **extra_feats}.items():
            if feat not in df.columns:
                continue
            series = df[feat]
            for feat_val, delta in (value_map or {}).items():
                if delta is None:
                    continue  # explicit "ignore" (null in YAML)
                mask = _cond_mask(series, feat_val)
                if mask.any():
                    z[mask.values] += float(delta)

        # 2) Optional continuity bonus from discrete decision (keeps behavior stable)
        if mode_col in df.columns:
            cur = df[mode_col].astype(str).str.strip().str.lower()
            for klass, bonus in base_bonus.items():
                if bonus is None:
                    continue  # null => don't nudge
                z[cur == str(klass).strip().lower()] += float(bonus)

        # 3) Mask special cases:
        #    - mangrove rows: not part of remote/field axis
        #    - EV "not available": don't force field bias; keep as NaN
        #    - "<mode> review required": ambiguous/erroneous -> route to manual review
        target = df.get("target_sys", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        dec    = df.get(mode_col, pd.Series("", index=df.index)).astype(str).str.strip().str.lower()

        mang_mask   = (target == "mangrove")
        review_mask = (dec == "review required")
        na_ev_mask  = (mode_col == ev_col) & (dec == "not available")

        #expose handy flags for QA/filters
        df[f"{prefix}mask_mangrove"]        = mang_mask
        df[f"{prefix}mask_not_available"]   = na_ev_mask
        df[f"{prefix}mask_review_required"] = review_mask

        # apply the masks to z
        z = pd.Series(z, index=df.index, name=f"{prefix}_raw")
        z[mang_mask | review_mask | na_ev_mask] = np.nan

        # 4) Map to 0..100 and build area-weighted score for roll-ups
        score_0_100 = pd.Series(_map_z_to_0_100(z.fillna(0).to_numpy(), scale_cfg),
                                index=df.index, name=f"{prefix}_score")
        score_0_100[z.isna()] = np.nan  # restore NaNs for masked rows

        area = pd.to_numeric(df.get(area_col, 1.0), errors="coerce").fillna(0).clip(lower=0)
        score_wtd = (score_0_100 * area).rename(f"{prefix}_score_wtd")

        # 5) Simple label via thresholds (purely for readability)
        #    Start with remote/field bins, then overwrite masked cases with special labels.
        bins = np.array([0, 30, 55, 70, 100], dtype=float)
        labels = np.array(["strong field", "weak field", "weak remote", "strong remote"])
        lab = pd.cut(score_0_100, bins=bins, labels=labels, include_lowest=True).astype(object)

        # overwrite labels for special cases (these rows have NaN scores by design)
        lab[mang_mask]   = "mangrove"
        lab[na_ev_mask]  = "not available"
        lab[review_mask] = "review required"

        # 6) Write columns
        df[z.name] = z
        df[score_0_100.name] = score_0_100.round(1)
        df[score_wtd.name]   = score_wtd.round(1)
        df[f"{prefix}_simple_bin"] = lab

    # Compute both modes in one pass
    _z_for(baseline_col, base_only, prefix="baseline_")
    _z_for(ev_col, ev_only, prefix="ev_")

    return df


def _label_from_score(score: float, thr: Dict[str, float]) -> Optional[str]:
    """
    Map a 0–100 Remote Suitability score to a simple label.
    """
    if pd.isna(score):
        return None
    if score >= thr.get("strong_remote", 70): return "strong remote"
    if score >= thr.get("weak_remote", 55):   return "weak remote"
    if score >= thr.get("weak_field", 30):    return "weak field"
    return "strong field"

def aggregate_project_score(
    params: Dict,
    df: pd.DataFrame,
    project_col: str = "project_id",
    area_col: Optional[str] = None,
    baseline_score_col: str = "baseline__score",
    ev_score_col: str = "ev__score",
    project_name_col: str = "project_name",
) -> pd.DataFrame:
    """
    Compute a single, area-weighted Remote Suitability score per PROJECT.

    Project score per mode (baseline/EV) is:
        sum(score_0_100 * area) / sum(area)   over polygons with a valid score.

    If the fraction of project area with valid polygon scores ("coverage")
    is below `min_coverage` (default 0.6), the project score is withheld
    as NaN and labeled "not available".
    """
    suit = (params or {}).get("suitability", {})
    area_col = area_col or suit.get("area_col", "area")
    min_cov = float(suit.get("min_coverage", 0.6))
    thr = suit.get("thresholds", {"weak_field": 30, "weak_remote": 55, "strong_remote": 70})

    # Clean area and prep working copy
    area = pd.to_numeric(df.get(area_col, 1.0), errors="coerce").fillna(0).clip(lower=0)
    work = df.copy()
    work["_area_clean_"] = area

    out_rows: List[Dict] = []
    for prj, g in work.groupby(project_col, dropna=False):
        total_area = g["_area_clean_"].sum()
        total_polys = len(g)

        # --- project_name: assume 1:1 mapping with project_id; take first non-null
        if project_name_col in g.columns and g[project_name_col].notna().any():
            prj_name = g[project_name_col].dropna().iloc[0]
        else:
            prj_name = None

        # --- Baseline aggregation ---
        bmask = g[baseline_score_col].notna()
        b_area = g.loc[bmask, "_area_clean_"].sum()
        b_cov = (b_area / total_area) if total_area > 0 else 0.0
        b_score = (
            (g.loc[bmask, baseline_score_col] * g.loc[bmask, "_area_clean_"]).sum() / b_area
            if b_area > 0 else np.nan
        )
        if b_cov < min_cov:
            b_score_pub, b_label = np.nan, "not available"
        else:
            b_score_pub = round(float(b_score), 1) if pd.notna(b_score) else np.nan
            b_label = _label_from_score(b_score_pub, thr) if pd.notna(b_score_pub) else "not available"

        # --- EV aggregation ---
        emask = g[ev_score_col].notna()
        e_area = g.loc[emask, "_area_clean_"].sum()
        e_cov = (e_area / total_area) if total_area > 0 else 0.0
        e_score = (
            (g.loc[emask, ev_score_col] * g.loc[emask, "_area_clean_"]).sum() / e_area
            if e_area > 0 else np.nan
        )
        if e_cov < min_cov:
            e_score_pub, e_label = np.nan, "not available"
        else:
            e_score_pub = round(float(e_score), 1) if pd.notna(e_score) else np.nan
            e_label = _label_from_score(e_score_pub, thr) if pd.notna(e_score_pub) else "not available"

        out_rows.append({
            project_col: prj,
            project_name_col: prj_name,           
            "total_area": round(float(total_area), 3),

            # Baseline
            "baseline_project_score_0_100": b_score_pub,
            "baseline_project_label": b_label,
            "baseline_coverage_area_frac": round(float(b_cov), 3),
            "baseline_scored_poly_count": int(bmask.sum()),
            "baseline_total_poly_count": int(total_polys),
            "scored_area_baseline": round(float(b_area), 3),

            # EV
            "ev_project_score_0_100": e_score_pub,
            "ev_project_label": e_label,
            "ev_coverage_area_frac": round(float(e_cov), 3),
            "ev_scored_poly_count": int(emask.sum()),
            "ev_total_poly_count": int(total_polys),
            "scored_area_ev": round(float(e_area), 3),
        })

    return pd.DataFrame(out_rows)