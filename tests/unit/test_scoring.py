"""
Unit tests for apply_scoring() in polygon_decisions.py.

Covers:
  - Weighted combination of canopy, imagery and slope components
  - Dynamic re-weighting when a component is missing
  - NaN score when every component is missing
  - Extended-window imagery weighting
  - Clipping of out-of-range inputs
  - NaN scores for mangrove / review required / not available / missing decisions
  - Missing-component flags and area-weighted columns
"""

import numpy as np
import pandas as pd
import pytest

from decision_tree.polygon_decisions import (
    apply_scoring,
    TCC_BASELINE_OFFSET_YEARS,
    TCC_EI_OFFSET_YEARS,
)

PARAMS = {
    "scoring": {
        "area_col": "area",
        "weight_canopy": 0.40,
        "weight_img": 0.40,
        "weight_slope": 0.20,
        "max_imgs": 4,
        "img_1yr_weight": 1.0,
        "img_ext_weight": 0.5,
    }
}

BASE_YEAR = 2022
BASE_TTC_COL = f"ttc_{BASE_YEAR + TCC_BASELINE_OFFSET_YEARS}"
EV_TTC_COL = f"ttc_{BASE_YEAR + TCC_EI_OFFSET_YEARS}"


def _poly(baseline_ttc=40.0, ev_ttc=40.0, img_1yr=4.0, img_ext=4.0, img_ev=4.0, slope_area=50.0,
          decision="weak remote", target_sys="agroforest", area=2.0):
    row = {
        "project_id": "p1",
        "poly_id": "poly_1",
        "target_sys": target_sys,
        "area": area,
        "baseline_year": BASE_YEAR,
        "baseline_img_count": img_1yr,
        "baseline_ext_img_count": img_ext,
        "ev_img_count": img_ev,
        "slope_area": slope_area,
        "baseline_decision": decision,
        "ev_decision": decision,
    }
    row[BASE_TTC_COL] = baseline_ttc
    if EV_TTC_COL != BASE_TTC_COL:
        row[EV_TTC_COL] = ev_ttc
    return row


def _score(**kwargs) -> pd.Series:
    return apply_scoring(PARAMS, pd.DataFrame([_poly(**kwargs)])).iloc[0]


class TestComponentWeighting:
    def test_all_components_present(self):
        # canopy 100-40=60; img (1*1 + 0.5*0)/1.5*100 = 66.67; slope 50
        # 0.4*60 + 0.4*66.67 + 0.2*50 = 60.67
        assert _score()["baseline_remote_suitability"] == pytest.approx(60.7)

    def test_missing_slope_is_reweighted(self):
        # (0.4*60 + 0.4*66.67) / 0.8 = 63.33
        res = _score(slope_area=np.nan)
        assert res["baseline_remote_suitability"] == pytest.approx(63.3)
        assert bool(res["slope_missing"]) is True

    def test_missing_canopy_is_reweighted(self):
        # (0.4*66.67 + 0.2*50) / 0.6 = 61.11
        res = _score(baseline_ttc=np.nan)
        assert res["baseline_remote_suitability"] == pytest.approx(61.1)
        assert bool(res["baseline_canopy_missing"]) is True

    def test_all_components_missing_gives_nan(self):
        res = _score(baseline_ttc=np.nan, ev_ttc=np.nan, img_1yr=np.nan, img_ext=np.nan,
                     img_ev=np.nan, slope_area=np.nan)
        assert np.isnan(res["baseline_remote_suitability"])
        assert np.isnan(res["ev_remote_suitability"])

    def test_extended_window_images_score_at_reduced_weight(self):
        # only imagery present: (1*0 + 0.5*1) / 1.5 * 100 = 33.33
        res = _score(baseline_ttc=np.nan, img_1yr=0.0, img_ext=4.0, slope_area=np.nan)
        assert res["baseline_remote_suitability"] == pytest.approx(33.3)

    def test_image_score_caps_at_max_imgs(self):
        # only imagery present: 1yr capped at 1.0, ext-only = 0 -> 66.67
        res = _score(baseline_ttc=np.nan, img_1yr=10.0, img_ext=10.0, slope_area=np.nan)
        assert res["baseline_remote_suitability"] == pytest.approx(66.7)

    def test_ev_uses_ev_imagery_and_ev_canopy_year(self):
        # canopy 100-10=90; img 2/4*100 = 50; slope 50 -> 0.4*90 + 0.4*50 + 0.2*50 = 66
        if EV_TTC_COL == BASE_TTC_COL:
            pytest.skip("baseline and EV canopy years coincide")
        res = _score(ev_ttc=10.0, img_ev=2.0)
        assert res["ev_remote_suitability"] == pytest.approx(66.0)


class TestClipping:
    def test_canopy_above_100_clips_to_zero_score(self):
        res = _score(baseline_ttc=120.0, img_1yr=np.nan, img_ext=np.nan, slope_area=np.nan)
        assert res["baseline_remote_suitability"] == pytest.approx(0.0)

    def test_negative_image_counts_clip_to_zero(self):
        res = _score(baseline_ttc=np.nan, img_1yr=-3.0, img_ext=-3.0, slope_area=np.nan)
        assert res["baseline_remote_suitability"] == pytest.approx(0.0)

    def test_slope_above_100_clips(self):
        res = _score(baseline_ttc=np.nan, img_1yr=np.nan, img_ext=np.nan, slope_area=150.0)
        assert res["baseline_remote_suitability"] == pytest.approx(100.0)


class TestUnscoreablePolygons:
    @pytest.mark.parametrize("decision", ["review required", "not available", None])
    def test_unscoreable_decisions_get_nan(self, decision):
        res = _score(decision=decision)
        assert np.isnan(res["baseline_remote_suitability"])
        assert np.isnan(res["ev_remote_suitability"])

    def test_mangrove_target_gets_nan(self):
        res = _score(target_sys="mangrove")
        assert np.isnan(res["baseline_remote_suitability"])
        assert np.isnan(res["ev_remote_suitability"])

    def test_decision_matching_is_case_and_whitespace_insensitive(self):
        assert np.isnan(_score(decision="  Review Required ")["baseline_remote_suitability"])


class TestOutputColumns:
    def test_area_weighted_score(self):
        res = _score(area=2.0)
        assert res["baseline_remote_suitability_wtd"] == pytest.approx(round(60.7 * 2.0, 1))

    def test_missing_flags_written(self):
        res = _score()
        for col in ["baseline_canopy_missing", "ev_canopy_missing", "baseline_img_missing",
                    "ev_img_missing", "slope_missing"]:
            assert bool(res[col]) is False

    def test_missing_extended_count_flags_baseline_imagery(self):
        assert bool(_score(img_ext=np.nan)["baseline_img_missing"]) is True
