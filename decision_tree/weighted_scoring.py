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
    Compute a single "Remote Suitability" score per polygon for baseline and EV.

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
        """
        Build and write the suitability score columns for one mode (baseline or EV).

        Steps:
          1. Feature contributions — iterate over all features defined in params
             (common + mode-specific). For each feature value/condition that matches
             a row, add the configured delta to that row's raw z score.
             z > 0 pushes toward remote; z < 0 pushes toward field.

          2. Discrete bonus — apply a small continuity nudge based on the
             classification decision already assigned (e.g. 'strong remote' gets +1.0).
             This keeps the continuous score consistent with the discrete decision.

          3. Masking — set z to NaN for polygons that should not receive a numeric
             score: mangrove polygons, 'review required', and 'not available'.
             QA flag columns (e.g. baseline_mask_mangrove) are written for traceability.

          4. Scale to 0–100 — pass z through logistic (or minmax) transformation.
             Area-weighted score is also produced (score × polygon area).

          5. Simple bin label — bucket the 0–100 score into human-readable labels
             (strong remote / weak remote / weak field / strong field), then
             overwrite masked polygons with their special label.

        Writes columns: {prefix}_raw, {prefix}_score, {prefix}_score_wtd,
                        {prefix}_simple_bin, {prefix}_mask_mangrove,
                        {prefix}_mask_review_required, {prefix}_mask_not_available
        """
        z = np.zeros(len(df), dtype=float)

        # 1) Feature contributions
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

        # 2) Discrete continuity bonus from classification decision
        if mode_col in df.columns:
            cur = df[mode_col].astype(str).str.strip().str.lower()
            for klass, bonus in base_bonus.items():
                if bonus is None:
                    continue
                z[cur == str(klass).strip().lower()] += float(bonus)

        # 3) Mask special cases — mangrove, review required, not available
        target = df.get("target_sys", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        dec    = df.get(mode_col, pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        mang_mask   = (target == "mangrove")
        review_mask = (dec == "review required")
        na_mask     = (dec == "not available")

        # QA flag columns for traceability
        df[f"{prefix}_mask_mangrove"]        = mang_mask
        df[f"{prefix}_mask_review_required"] = review_mask
        df[f"{prefix}_mask_not_available"]   = na_mask

        z = pd.Series(z, index=df.index, name=f"{prefix}_raw")
        z[mang_mask | review_mask | na_mask] = np.nan

        # 4) Scale to 0..100 and area-weight
        score_0_100 = pd.Series(
            _map_z_to_0_100(z.fillna(0).to_numpy(), scale_cfg),
            index=df.index, name=f"{prefix}_score"
        )
        score_0_100[z.isna()] = np.nan
        area = pd.to_numeric(df.get(area_col, 1.0), errors="coerce").fillna(0).clip(lower=0)
        score_wtd = (score_0_100 * area).rename(f"{prefix}_score_wtd")

        # 5) Simple bin label; overwrite masked cases with their special label
        bins = np.array([0, 30, 55, 70, 100], dtype=float)
        labels = np.array(["strong field", "weak field", "weak remote", "strong remote"])
        lab = pd.cut(score_0_100, bins=bins, labels=labels, include_lowest=True).astype(object)
        lab[mang_mask]   = "mangrove"
        lab[na_mask]     = "not available"
        lab[review_mask] = "review required"

        # Write outputs
        df[z.name] = z
        df[score_0_100.name] = score_0_100.round(1)
        df[score_wtd.name]   = score_wtd.round(1)
        df[f"{prefix}_simple_bin"] = lab

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


def _score_one_mode(
    g: pd.DataFrame,
    score_col: str,
    mask_prefix: str,
    total_area: float,
    min_cov: float,
    thr: dict,
    mang_thr: float,
    rev_thr: float,
    precedence: list,
) -> dict:
    """
    Compute score, label, and coverage stats for one mode (baseline or EV)
    for a single project group.

    Parameters
    ----------
    g            : polygon-level rows for one project
    score_col    : name of the 0–100 score column (e.g. 'baseline_score')
    mask_prefix  : prefix used on QA mask columns (e.g. 'baseline')
    total_area   : sum of all polygon areas in the project
    min_cov      : minimum fraction of area that must be scored to publish a result
    thr          : score thresholds for labelling (weak_field, weak_remote, strong_remote)
    mang_thr     : area fraction above which project is labelled 'mangrove'
    rev_thr      : area fraction above which project is labelled 'review required'
    precedence   : order in which special labels override the numeric label

    Returns
    -------
    dict with keys:
      score_pub        : area-weighted project score (NaN if coverage too low)
      label            : human-readable label (strong remote … strong field,
                         or mangrove / review required / not available)
      pct_area_scored  : fraction of project area with a valid score (0–1)
      poly_frac_scored : fraction of polygons with a valid score (0–1)
      scored_area      : total area of scored polygons
      frac_mangrove    : fraction of project area flagged as mangrove
    """
    scored_mask = g[score_col].notna()
    area_scored = g.loc[scored_mask, "_area_clean_"].sum()
    coverage    = (area_scored / total_area) if total_area > 0 else 0.0

    mang_mask = g.get(f"{mask_prefix}_mask_mangrove",
                      pd.Series(False, index=g.index)).astype(bool)
    rev_mask  = g.get(f"{mask_prefix}_mask_review_required",
                      pd.Series(False, index=g.index)).astype(bool)

    frac_mang = g.loc[mang_mask, "_area_clean_"].sum() / total_area if total_area > 0 else 0.0
    frac_rev  = g.loc[rev_mask,  "_area_clean_"].sum() / total_area if total_area > 0 else 0.0

    poly_frac_scored = round(scored_mask.sum() / len(g), 3) if len(g) > 0 else 0.0

    raw_score = (
        (g.loc[scored_mask, score_col] * g.loc[scored_mask, "_area_clean_"]).sum() / area_scored
        if area_scored > 0 else np.nan
    )

    # Provisional label from coverage
    if coverage < min_cov:
        score_pub, label = np.nan, "not available"
    else:
        score_pub = round(float(raw_score), 1) if pd.notna(raw_score) else np.nan
        label = _label_from_score(score_pub, thr) if pd.notna(score_pub) else "not available"

    # Special-label overrides in precedence order
    for tag in precedence:
        if tag == "mangrove"        and frac_mang >= mang_thr: label, score_pub = "mangrove", np.nan; break
        if tag == "review required" and frac_rev  >= rev_thr:  label, score_pub = "review required", np.nan; break
        if tag == "not available"   and coverage  < min_cov:   label, score_pub = "not available", np.nan; break

    return {
        "score_pub":        score_pub,
        "label":            label,
        "pct_area_scored":  round(float(coverage), 3),
        "poly_frac_scored": poly_frac_scored,
        "scored_area":      round(float(area_scored), 3),
        "frac_mangrove":    round(float(frac_mang), 3),
    }


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

    Output columns
    --------------
    base_score            : area-weighted 0–100 suitability score
    base_label            : human-readable label for the project
    base_pct_area_scored  : fraction of project area with a valid polygon score (0–1)
    base_poly_frac_scored : fraction of polygons with a valid score (0–1)
    base_scored_area      : total area (ha) of scored polygons
    base_frac_mangrove    : fraction of project area flagged as mangrove
    ev_*                  : same set of columns for the EV mode
    """
    suit = (params or {}).get("suitability", {})
    area_col = area_col or suit.get("area_col", "area")
    min_cov = float(suit.get("min_coverage", 0.6))
    thr = suit.get("thresholds", {"weak_field": 30, "weak_remote": 55, "strong_remote": 70})

    spec = suit.get("special_labels", {}) or {}
    mang_thr   = float(spec.get("mangrove_area_frac_threshold", 0.10))
    rev_thr    = float(spec.get("review_area_frac_threshold", 0.10))
    precedence = spec.get("precedence", ["mangrove", "review required", "not available"])

    area = pd.to_numeric(df.get(area_col, 1.0), errors="coerce").fillna(0).clip(lower=0)
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

        b = _score_one_mode(g, baseline_score_col, "baseline",
                            total_area, min_cov, thr, mang_thr, rev_thr, precedence)
        e = _score_one_mode(g, ev_score_col, "ev",
                            total_area, min_cov, thr, mang_thr, rev_thr, precedence)

        out_rows.append({
            project_col:      prj,
            project_name_col: prj_name,
            "total_area":     round(float(total_area), 3),

            # Baseline
            "base_score":            b["score_pub"],
            "base_label":            b["label"],
            "base_pct_area_scored":  b["pct_area_scored"],
            "base_poly_frac_scored": b["poly_frac_scored"],
            "base_scored_area":      b["scored_area"],
            "base_frac_mangrove":    b["frac_mangrove"],

            # EV
            "ev_score":            e["score_pub"],
            "ev_label":            e["label"],
            "ev_pct_area_scored":  e["pct_area_scored"],
            "ev_poly_frac_scored": e["poly_frac_scored"],
            "ev_scored_area":      e["scored_area"],
            "ev_frac_mangrove":    e["frac_mangrove"],
        })

    return pd.DataFrame(out_rows)