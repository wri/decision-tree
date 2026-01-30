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

    Thresholds
    ----------
    The simple label uses these defaults on 0..100 (tune if desired):
      >= 70  => "strong remote"
      55-70  => "weak remote"
      30-55  => "weak field"
      <  30  => "strong field"
    """
    cfg = (params or {}).get("suitability", {})
    area_col  = cfg.get("area_col", "area")
    scale_cfg = cfg.get("scale", {})
    base_bonus = cfg.get("base_bonus", {}) or {}

    feats     = cfg.get("features", {}) or {}
    common    = feats.get("common", {}) or {}
    base_only = feats.get("baseline_only", {}) or {}
    ev_only   = feats.get("ev_only", {}) or {}

    def _z_for(mode_col: str, extra_feats: dict, prefix: str):
        z = np.zeros(len(df), dtype=float)

        # 1) feature contributions
        for feat, value_map in {**common, **extra_feats}.items():
            if feat not in df.columns:
                continue
            series = df[feat]
            for feat_val, delta in (value_map or {}).items():
                if delta is None:
                    continue
                mask = _cond_mask(series, feat_val)
                if mask.any():
                    z[mask.values] += float(delta)

        # 2) optional continuity bonus from discrete decision
        if mode_col in df.columns:
            cur = df[mode_col].astype(str).str.strip().str.lower()
            for klass, bonus in base_bonus.items():
                if bonus is None:
                    continue
                z[cur == str(klass).strip().lower()] += float(bonus)

        # 3) Mask special cases (mode-aware)
        #    - Mask mangrove & review in both modes
        #    - Mask "not available" in whichever mode we're scoring (baseline OR EV)
        target = df.get("target_sys", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        dec    = df.get(mode_col, pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        mang_mask   = (target == "mangrove")
        review_mask = (dec == "review required")
        na_mask     = (dec == "not available")   # <-- mode-aware: baseline NA and EV NA both masked

        # QA flags
        df[f"{prefix}_mask_mangrove"]        = mang_mask
        df[f"{prefix}_mask_review_required"] = review_mask
        df[f"{prefix}_mask_not_available"]   = na_mask

        # (optional) reason column helps debugging
        # reason = pd.Series("", index=df.index)
        # reason[mang_mask] = "mangrove"; reason[review_mask] = "review required"; reason[na_mask] = "not available"
        # df[f"{prefix}_mask_reason"] = reason.replace("", np.nan)

        # Apply masks to raw axis
        z = pd.Series(z, index=df.index, name=f"{prefix}_raw")
        z[mang_mask | review_mask | na_mask] = np.nan

        # 4) map to 0..100 and area-weight
        score_0_100 = pd.Series(
            _map_z_to_0_100(z.fillna(0).to_numpy(), scale_cfg),
            index=df.index, name=f"{prefix}_score"
        )
        score_0_100[z.isna()] = np.nan
        area = pd.to_numeric(df.get(area_col, 1.0), errors="coerce").fillna(0).clip(lower=0)
        score_wtd = (score_0_100 * area).rename(f"{prefix}_score_wtd")

        # 5) simple label for readability (remote/field bins); overwrite masked cases
        bins = np.array([0, 30, 55, 70, 100], dtype=float)
        labels = np.array(["strong field", "weak field", "weak remote", "strong remote"])
        lab = pd.cut(score_0_100, bins=bins, labels=labels, include_lowest=True).astype(object)
        lab[mang_mask] = "mangrove"
        lab[na_mask]   = "not available"
        lab[review_mask] = "review required"

        # write outputs
        df[z.name] = z
        df[score_0_100.name] = score_0_100.round(1)
        df[score_wtd.name]   = score_wtd.round(1)
        df[f"{prefix}_simple_bin"] = lab

    # single-underscore prefixes (prevents double-underscore column names)
    _z_for(baseline_col, base_only, prefix="baseline")
    _z_for(ev_col,       ev_only,   prefix="ev")

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
    baseline_score_col: str = "baseline_score",
    ev_score_col: str = "ev_score",
    project_name_col: str = "project_name",
) -> pd.DataFrame:
    """
    Compute a single, area-weighted Remote Suitability score per PROJECT.

    - Project score per mode (baseline/EV):
          sum(score_0_100 * area) / sum(area)   over polygons with a valid score.
    - If scored-area coverage < min_coverage, publish NaN and label 'not available'.
    - Special labels 'mangrove' / 'review required' can override via area fractions.

    'not available' is determined *only* by coverage (no NA-area threshold).
    """
    suit = (params or {}).get("suitability", {})
    area_col = area_col or suit.get("area_col", "area")
    min_cov = float(suit.get("min_coverage", 0.6))
    thr = suit.get("thresholds", {"weak_field": 30, "weak_remote": 55, "strong_remote": 70})

    # Special-label thresholds & precedence (no NA threshold)
    spec = suit.get("special_labels", {}) or {}
    mang_thr = float(spec.get("mangrove_area_frac_threshold", 0.10))
    rev_thr  = float(spec.get("review_area_frac_threshold", 0.10))
    precedence = spec.get("precedence", ["mangrove", "review required", "not available"])

    # Clean area
    area = pd.to_numeric(df.get(area_col, 1.0), errors="coerce").fillna(0).clip(lower=0)
    work = df.copy()
    work["_area_clean_"] = area

    out_rows: List[Dict] = []
    for prj, g in work.groupby(project_col, dropna=False):
        total_area = g["_area_clean_"].sum()
        total_polys = len(g)

        # Project name (first non-null)
        prj_name = g[project_name_col].dropna().iloc[0] if project_name_col in g.columns and g[project_name_col].notna().any() else None

        # ---------- BASELINE ----------
        bmask_scored = g[baseline_score_col].notna()
        b_area_scored = g.loc[bmask_scored, "_area_clean_"].sum()
        b_cov = (b_area_scored / total_area) if total_area > 0 else 0.0

        # Special-label area fractions (mangrove / review)
        b_mang = (g.get("baseline_mask_mangrove", pd.Series(False, index=g.index))).astype(bool)
        b_rev  = (g.get("baseline_mask_review_required", pd.Series(False, index=g.index))).astype(bool)

        b_area_mang = g.loc[b_mang, "_area_clean_"].sum()
        b_area_rev  = g.loc[b_rev,  "_area_clean_"].sum()
        b_frac_mang = (b_area_mang / total_area) if total_area > 0 else 0.0
        b_frac_rev  = (b_area_rev  / total_area) if total_area > 0 else 0.0

        # Numeric score (area-weighted over scored polygons)
        if b_area_scored > 0:
            b_score = (g.loc[bmask_scored, baseline_score_col] * g.loc[bmask_scored, "_area_clean_"]).sum() / b_area_scored
        else:
            b_score = np.nan

        # Provisional publish via coverage
        if b_cov < min_cov:
            b_score_pub = np.nan
            b_label = "not available"
        else:
            b_score_pub = round(float(b_score), 1) if pd.notna(b_score) else np.nan
            b_label = _label_from_score(b_score_pub, thr) if pd.notna(b_score_pub) else "not available"

        # Override by precedence (no NA-area branch)
        for tag in precedence:
            if tag == "mangrove" and b_frac_mang >= mang_thr:
                b_label, b_score_pub = "mangrove", np.nan
                break
            if tag == "review required" and b_frac_rev >= rev_thr:
                b_label, b_score_pub = "review required", np.nan
                break
            if tag == "not available" and b_cov < min_cov:
                b_label, b_score_pub = "not available", np.nan
                break

        # ---------- EV ----------
        emask_scored = g[ev_score_col].notna()
        e_area_scored = g.loc[emask_scored, "_area_clean_"].sum()
        e_cov = (e_area_scored / total_area) if total_area > 0 else 0.0

        e_mang = (g.get("ev_mask_mangrove", pd.Series(False, index=g.index))).astype(bool)
        e_rev  = (g.get("ev_mask_review_required", pd.Series(False, index=g.index))).astype(bool)

        e_area_mang = g.loc[e_mang, "_area_clean_"].sum()
        e_area_rev  = g.loc[e_rev,  "_area_clean_"].sum()
        e_frac_mang = (e_area_mang / total_area) if total_area > 0 else 0.0
        e_frac_rev  = (e_area_rev  / total_area) if total_area > 0 else 0.0

        if e_area_scored > 0:
            e_score = (g.loc[emask_scored, ev_score_col] * g.loc[emask_scored, "_area_clean_"]).sum() / e_area_scored
        else:
            e_score = np.nan

        if e_cov < min_cov:
            e_score_pub = np.nan
            e_label = "not available"
        else:
            e_score_pub = round(float(e_score), 1) if pd.notna(e_score) else np.nan
            e_label = _label_from_score(e_score_pub, thr) if pd.notna(e_score_pub) else "not available"

        for tag in precedence:
            if tag == "mangrove" and e_frac_mang >= mang_thr:
                e_label, e_score_pub = "mangrove", np.nan
                break
            if tag == "review required" and e_frac_rev >= rev_thr:
                e_label, e_score_pub = "review required", np.nan
                break
            if tag == "not available" and e_cov < min_cov:
                e_label, e_score_pub = "not available", np.nan
                break

        out_rows.append({
            project_col: prj,
            project_name_col: prj_name,
            "total_area": round(float(total_area), 3),

            # Baseline
            "baseline_project_score_0_100": b_score_pub,
            "baseline_project_label": b_label,
            "baseline_coverage_area_frac": round(float(b_cov), 3),
            "baseline_scored_poly_count": int(bmask_scored.sum()),
            "baseline_total_poly_count": int(total_polys),
            "scored_area_baseline": round(float(b_area_scored), 3),
            "baseline_frac_mangrove": round(float(b_frac_mang), 3),
            "baseline_frac_review":   round(float(b_frac_rev), 3),

            # EV
            "ev_project_score_0_100": e_score_pub,
            "ev_project_label": e_label,
            "ev_coverage_area_frac": round(float(e_cov), 3),
            "ev_scored_poly_count": int(emask_scored.sum()),
            "ev_total_poly_count": int(total_polys),
            "scored_area_ev": round(float(e_area_scored), 3),
            "ev_frac_mangrove": round(float(e_frac_mang), 3),
            "ev_frac_review":   round(float(e_frac_rev), 3),
        })

    return pd.DataFrame(out_rows)