"""Unit tests for calc_cost_to_verify() in cost_calculator.py."""

import numpy as np
import pandas as pd
import pytest

from decision_tree.constants import COST_FIELD, COST_REMOTE
from decision_tree.cost_calculator import calc_cost_to_verify


def _df(rows):
    return pd.DataFrame(rows, columns=["baseline_decision", "ev_decision", "area"])


@pytest.mark.parametrize("decision, rate", [
    ("weak remote", COST_REMOTE), ("strong remote", COST_REMOTE),
    ("weak field", COST_FIELD), ("strong field", COST_FIELD),
])
def test_rate_by_decision(decision, rate):
    out = calc_cost_to_verify(_df([[decision, decision, 10.0]]))
    assert out["baseline_cost"].iat[0] == pytest.approx(round(10.0 * rate, 1))
    assert out["ev_cost"].iat[0] == pytest.approx(round(10.0 * rate, 1))


@pytest.mark.parametrize("decision", ["mangrove", "not available", "review required"])
def test_zero_cost_decisions(decision):
    out = calc_cost_to_verify(_df([[decision, decision, 10.0]]))
    assert out["baseline_cost"].iat[0] == 0.0
    assert out["ev_cost"].iat[0] == 0.0


def test_baseline_and_ev_use_their_own_decision():
    out = calc_cost_to_verify(_df([["strong field", "weak remote", 10.0]]))
    assert out["baseline_cost"].iat[0] == pytest.approx(10.0 * COST_FIELD)
    assert out["ev_cost"].iat[0] == pytest.approx(round(10.0 * COST_REMOTE, 1))


def test_decision_matching_is_case_and_whitespace_insensitive():
    out = calc_cost_to_verify(_df([[" Strong Field ", "STRONG FIELD", 1.0]]))
    assert out["baseline_cost"].iat[0] == pytest.approx(COST_FIELD)
    assert out["ev_cost"].iat[0] == pytest.approx(COST_FIELD)


def test_unknown_decision_warns_and_costs_zero():
    with pytest.warns(UserWarning, match="Unrecognized baseline_decision"):
        out = calc_cost_to_verify(_df([["maybe remote", "weak remote", 10.0]]))
    assert out["baseline_cost"].iat[0] == 0.0


def test_missing_area_costs_zero():
    out = calc_cost_to_verify(_df([["strong field", "strong field", np.nan]]))
    assert out["baseline_cost"].iat[0] == 0.0


def test_costs_are_rounded():
    out = calc_cost_to_verify(_df([["weak remote", "weak remote", 1.234]]), decimals=1)
    assert out["baseline_cost"].iat[0] == pytest.approx(round(1.234 * COST_REMOTE, 1))


def test_input_is_modified_in_place():
    # Pins current behaviour: the input frame gains the cost columns.
    df = _df([["weak remote", "weak remote", 1.0]])
    out = calc_cost_to_verify(df)
    assert out is df
    assert "baseline_cost" in df.columns
