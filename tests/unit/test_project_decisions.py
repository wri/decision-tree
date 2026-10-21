"""
Unit tests for aggregate_project_score() in project_decisions.py.

Covers:
  - Area-weighted majority label and area-weighted confidence
  - Confidence inversion for field projects (and not for remote projects)
  - Tie-breaking between equal-area labels
  - Override priority: mangrove > review required > not available, at thresholds
  - min_coverage gate (polygon fraction today; area fraction per docs - xfail)
  - Exclusion of polygons with no decision or no area
  - Zero total area
  - Context-aware data quality flags (HIGH/LOW priority by project label)
"""

import numpy as np
import pandas as pd
import pytest

from decision_tree.project_decisions import aggregate_project_score

PARAMS = {
    "scoring": {
        "area_col": "area",
        "min_coverage": 0.60,
        "mangrove_frac_threshold": 0.20,
        "review_frac_threshold": 0.60,
        "missing_data_threshold": 0.20,
    }
}


def _poly(decision="weak remote", score=50.0, area=1.0, target_sys="agroforest",
          project_id="p1", **extra):
    """One polygon row with identical baseline and EV values unless overridden."""
    row = {
        "project_id": project_id,
        "project_name": f"name_{project_id}",
        "poly_id": None,
        "target_sys": target_sys,
        "area": area,
        "baseline_decision": decision,
        "ev_decision": decision,
        "baseline_remote_suitability": score,
        "ev_remote_suitability": score,
    }
    row.update(extra)
    return row


def _aggregate(rows, params=PARAMS):
    df = pd.DataFrame(rows)
    df["poly_id"] = [f"poly_{i}" for i in range(len(df))]
    return aggregate_project_score(params, df)


def _one(rows, params=PARAMS):
    out = _aggregate(rows, params)
    assert len(out) == 1
    return out.iloc[0]


# ---------------------------------------------------------------------------
# Label and confidence
# ---------------------------------------------------------------------------
class TestLabelAndConfidence:
    def test_majority_label_by_area_not_by_count(self):
        rows = [_poly("weak remote", 80.0, area=10.0),
                _poly("strong field", 20.0, area=3.0),
                _poly("strong field", 20.0, area=3.0)]
        res = _one(rows)
        assert res["base_label"] == "weak remote"
        assert res["ev_label"] == "weak remote"

    def test_confidence_is_area_weighted_mean(self):
        rows = [_poly("weak remote", 80.0, area=10.0),
                _poly("strong field", 20.0, area=3.0)]
        res = _one(rows)
        # (80*10 + 20*3) / 13 = 66.15
        assert res["base_conf"] == pytest.approx(66.2)

    def test_field_confidence_is_inverted(self):
        res = _one([_poly("strong field", 30.0)])
        assert res["base_label"] == "strong field"
        assert res["base_conf"] == pytest.approx(70.0)

    def test_remote_confidence_is_not_inverted(self):
        res = _one([_poly("weak remote", 30.0)])
        assert res["base_conf"] == pytest.approx(30.0)

    def test_equal_area_tie_takes_first_label_alphabetically(self):
        # Pins current behaviour: groupby sorts labels and idxmax returns the first max.
        rows = [_poly("weak remote", 80.0, area=5.0),
                _poly("strong field", 20.0, area=5.0)]
        assert _one(rows)["base_label"] == "strong field"

    def test_one_row_per_project_with_name(self):
        rows = [_poly(project_id="p1"), _poly(project_id="p1"), _poly(project_id="p2")]
        out = _aggregate(rows).set_index("project_id")
        assert sorted(out.index) == ["p1", "p2"]
        assert out.loc["p2", "project_name"] == "name_p2"
        assert out.loc["p1", "total_area"] == pytest.approx(2.0)

    def test_baseline_and_ev_are_scored_independently(self):
        rows = [_poly("weak remote", 80.0, ev_decision="strong field", ev_remote_suitability=10.0)]
        res = _one(rows)
        assert res["base_label"] == "weak remote"
        assert res["ev_label"] == "strong field"
        assert res["ev_conf"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Override priority and thresholds
# ---------------------------------------------------------------------------
class TestOverrides:
    def test_mangrove_at_threshold_overrides(self):
        # mangrove area 2 of 10 = 0.20 (threshold is >=)
        rows = [_poly("mangrove", np.nan, area=2.0, target_sys="mangrove")] + \
               [_poly("weak remote", 80.0, area=2.0) for _ in range(4)]
        res = _one(rows)
        assert res["base_label"] == "mangrove"
        assert np.isnan(res["base_conf"])
        assert res["base_frac_mangrove"] == pytest.approx(0.2)

    def test_mangrove_below_threshold_does_not_override(self):
        rows = [_poly("mangrove", np.nan, area=1.9, target_sys="mangrove")] + \
               [_poly("weak remote", 80.0, area=2.0) for _ in range(4)]
        assert _one(rows)["base_label"] == "weak remote"

    def test_mangrove_takes_priority_over_review_required(self):
        rows = [_poly("mangrove", np.nan, area=5.0, target_sys="mangrove")] + \
               [_poly("review required", np.nan, area=1.0) for _ in range(4)]
        assert _one(rows)["base_label"] == "mangrove"

    def test_review_required_at_threshold_overrides(self):
        # 3 of 5 polygons review required = 0.60 (threshold is >=)
        rows = [_poly("review required", np.nan) for _ in range(3)] + \
               [_poly("weak remote", 80.0) for _ in range(2)]
        res = _one(rows)
        assert res["base_label"] == "review required"
        assert np.isnan(res["base_conf"])

    def test_review_required_below_threshold_does_not_override(self):
        rows = [_poly("review required", np.nan) for _ in range(2)] + \
               [_poly("weak remote", 80.0) for _ in range(3)]
        assert _one(rows)["base_label"] == "weak remote"

    def test_review_fraction_uses_polygon_count_not_area(self):
        # One huge review polygon must not override on its own.
        rows = [_poly("review required", np.nan, area=100.0)] + \
               [_poly("weak remote", 80.0, area=1.0) for _ in range(4)]
        assert _one(rows)["base_label"] != "review required"


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------
class TestCoverageGate:
    def test_polygon_fraction_at_min_coverage_is_published(self):
        rows = [_poly("weak remote", 80.0) for _ in range(3)] + \
               [_poly("weak remote", np.nan) for _ in range(2)]
        res = _one(rows)
        assert res["base_label"] == "weak remote"
        assert res["base_poly_frac_scored"] == pytest.approx(0.6)

    def test_polygon_fraction_below_min_coverage_is_not_available(self):
        rows = [_poly("weak remote", 80.0) for _ in range(2)] + \
               [_poly("weak remote", np.nan) for _ in range(3)]
        res = _one(rows)
        assert res["base_label"] == "not available"
        assert np.isnan(res["base_conf"])

    @pytest.mark.xfail(strict=True, reason=(
        "params.yaml and the docstring describe min_coverage as the fraction of project AREA "
        "with a valid score, but _score_one_mode gates on the fraction of POLYGONS scored. "
        "Remove this marker once the intended behaviour is confirmed and implemented."))
    def test_min_coverage_is_applied_to_area_fraction(self):
        # 4 of 5 polygons scored (0.8) but only 4 of 104 ha scored (~0.04)
        rows = [_poly("weak remote", np.nan, area=100.0)] + \
               [_poly("weak remote", 80.0, area=1.0) for _ in range(4)]
        assert _one(rows)["base_label"] == "not available"


# ---------------------------------------------------------------------------
# Exclusions and edge cases
# ---------------------------------------------------------------------------
class TestExclusions:
    def test_polygon_without_decision_is_not_scored(self):
        rows = [_poly("weak remote", 80.0, area=2.0) for _ in range(4)] + \
               [_poly(None, 10.0, area=2.0)]
        res = _one(rows)
        assert res["base_scored_area"] == pytest.approx(8.0)
        assert res["base_poly_frac_scored"] == pytest.approx(0.8)
        assert res["base_conf"] == pytest.approx(80.0)

    def test_polygon_without_area_is_excluded_from_totals(self):
        rows = [_poly("weak remote", 80.0, area=2.0) for _ in range(4)] + \
               [_poly("weak remote", 10.0, area=np.nan)]
        res = _one(rows)
        assert res["total_area"] == pytest.approx(8.0)
        assert res["base_conf"] == pytest.approx(80.0)

    def test_zero_total_area_does_not_raise(self):
        res = _one([_poly("weak remote", 80.0, area=0.0) for _ in range(3)])
        assert res["base_pct_area_scored"] == 0.0
        assert res["base_frac_mangrove"] == 0.0
        assert np.isnan(res["base_conf"])


# ---------------------------------------------------------------------------
# Data quality flags
# ---------------------------------------------------------------------------
def _with_missing(decision, score, n, n_missing, col):
    return [_poly(decision, score, **{col: i < n_missing}) for i in range(n)]


class TestDataQualityFlags:
    def test_field_project_slope_missing_is_high(self):
        flag = _one(_with_missing("strong field", 20.0, 5, 1, "slope_missing"))["base_data_quality_flag"]
        assert flag == "slope missing for 20% of polygons [HIGH: field project]"

    def test_field_project_canopy_missing_is_low(self):
        flag = _one(_with_missing("strong field", 20.0, 5, 1, "baseline_canopy_missing"))["base_data_quality_flag"]
        assert flag == "canopy missing for 20% of polygons [LOW: field project]"

    def test_remote_project_imagery_missing_is_high(self):
        flag = _one(_with_missing("weak remote", 80.0, 5, 1, "baseline_img_missing"))["base_data_quality_flag"]
        assert flag == "imagery missing for 20% of polygons [HIGH: remote project]"

    def test_remote_project_slope_missing_is_low(self):
        flag = _one(_with_missing("weak remote", 80.0, 5, 1, "slope_missing"))["base_data_quality_flag"]
        assert flag == "slope missing for 20% of polygons [LOW: remote project]"

    def test_below_threshold_is_not_flagged(self):
        flag = _one(_with_missing("weak remote", 80.0, 10, 1, "slope_missing"))["base_data_quality_flag"]
        assert flag == ""

    def test_ev_flags_use_ev_columns(self):
        rows = _with_missing("weak remote", 80.0, 5, 1, "ev_canopy_missing")
        res = _one(rows)
        assert res["base_data_quality_flag"] == ""
        assert res["ev_data_quality_flag"] == "canopy missing for 20% of polygons [HIGH: remote project]"

    def test_non_field_non_remote_label_flags_without_priority(self):
        rows = [_poly("review required", np.nan, slope_missing=True) for _ in range(5)]
        flag = _one(rows)["base_data_quality_flag"]
        assert flag == "slope missing for 100% of polygons"
