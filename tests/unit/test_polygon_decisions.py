"""
Unit tests for parse_condition() in polygon_decisions.py.

Covers:
  - All five comparison operators (>=, <=, >, <, ==)
  - Boundary values (equal-to-threshold)
  - Negative thresholds
  - Float thresholds
  - Two-char operators not mis-parsed as single-char (e.g. ">=" not matched as ">")
  - Direct equality fallback for non-operator strings
  - Direct equality fallback for non-string values
  - ValueError raised for malformed condition strings
"""

import pytest
from decision_tree.polygon_decisions import parse_condition


# ---------------------------------------------------------------------------
# >= (greater than or equal)
# ---------------------------------------------------------------------------
class TestGreaterThanOrEqual:
    def test_equal_to_threshold_returns_true(self):
        assert parse_condition(">=1", 1) is True

    def test_above_threshold_returns_true(self):
        assert parse_condition(">=1", 2) is True

    def test_below_threshold_returns_false(self):
        assert parse_condition(">=1", 0) is False

    def test_boundary_zero(self):
        assert parse_condition(">=0", 0) is True

    def test_float_threshold_equal(self):
        assert parse_condition(">=1.5", 1.5) is True

    def test_float_threshold_above(self):
        assert parse_condition(">=1.5", 2.0) is True

    def test_float_threshold_below(self):
        assert parse_condition(">=1.5", 1.0) is False

    def test_negative_threshold_above(self):
        assert parse_condition(">=-1", 0) is True

    def test_negative_threshold_equal(self):
        assert parse_condition(">=-1", -1) is True

    def test_negative_threshold_below(self):
        assert parse_condition(">=-1", -2) is False

    def test_not_confused_with_gt(self):
        """>=1 with actual=1 must be True; if parsed as > it would be False."""
        assert parse_condition(">=1", 1) is True


# ---------------------------------------------------------------------------
# <= (less than or equal)
# ---------------------------------------------------------------------------
class TestLessThanOrEqual:
    def test_equal_to_threshold_returns_true(self):
        assert parse_condition("<=5", 5) is True

    def test_below_threshold_returns_true(self):
        assert parse_condition("<=5", 4) is True

    def test_above_threshold_returns_false(self):
        assert parse_condition("<=5", 6) is False

    def test_float_threshold(self):
        assert parse_condition("<=2.5", 2.5) is True
        assert parse_condition("<=2.5", 3.0) is False

    def test_not_confused_with_lt(self):
        """<=5 with actual=5 must be True; if parsed as < it would be False."""
        assert parse_condition("<=5", 5) is True


# ---------------------------------------------------------------------------
# > (strictly greater than)
# ---------------------------------------------------------------------------
class TestGreaterThan:
    def test_above_threshold_returns_true(self):
        assert parse_condition(">0", 1) is True

    def test_equal_to_threshold_returns_false(self):
        assert parse_condition(">0", 0) is False

    def test_below_threshold_returns_false(self):
        assert parse_condition(">0", -1) is False

    def test_float_threshold(self):
        assert parse_condition(">1.5", 1.6) is True
        assert parse_condition(">1.5", 1.5) is False


# ---------------------------------------------------------------------------
# < (strictly less than)
# ---------------------------------------------------------------------------
class TestLessThan:
    def test_below_threshold_returns_true(self):
        assert parse_condition("<10", 9) is True

    def test_equal_to_threshold_returns_false(self):
        assert parse_condition("<10", 10) is False

    def test_above_threshold_returns_false(self):
        assert parse_condition("<10", 11) is False

    def test_float_threshold(self):
        assert parse_condition("<2.5", 2.4) is True
        assert parse_condition("<2.5", 2.5) is False


# ---------------------------------------------------------------------------
# == (numeric equality via operator string)
# ---------------------------------------------------------------------------
class TestNumericEquality:
    def test_match_returns_true(self):
        assert parse_condition("==3", 3) is True

    def test_no_match_returns_false(self):
        assert parse_condition("==3", 4) is False

    def test_float_threshold(self):
        assert parse_condition("==2.0", 2.0) is True
        assert parse_condition("==2.0", 2.1) is False


# ---------------------------------------------------------------------------
# Direct equality fallback (no operator prefix)
# ---------------------------------------------------------------------------
class TestDirectEquality:
    def test_matching_strings_return_true(self):
        assert parse_condition("remote", "remote") is True

    def test_non_matching_strings_return_false(self):
        assert parse_condition("remote", "field") is False

    def test_matching_integers_return_true(self):
        assert parse_condition(1, 1) is True

    def test_non_matching_integers_return_false(self):
        assert parse_condition(2, 1) is False

    def test_matching_floats_return_true(self):
        assert parse_condition(1.5, 1.5) is True

    def test_string_that_contains_operator_not_at_start(self):
        """A string with an operator not at position 0 is treated as a plain value."""
        assert parse_condition("x>=1", "x>=1") is True
        assert parse_condition("x>=1", "other") is False


# ---------------------------------------------------------------------------
# Malformed condition strings → ValueError
# ---------------------------------------------------------------------------
class TestMalformedConditions:
    def test_operator_followed_by_letters_raises(self):
        with pytest.raises(ValueError, match="Malformed condition string"):
            parse_condition(">abc", 5)

    def test_operator_with_space_before_number_raises(self):
        with pytest.raises(ValueError, match="Malformed condition string"):
            parse_condition(">= 1", 5)

    def test_operator_with_trailing_chars_raises(self):
        with pytest.raises(ValueError, match="Malformed condition string"):
            parse_condition(">=1x", 5)

    def test_operator_only_no_number_raises(self):
        with pytest.raises(ValueError, match="Malformed condition string"):
            parse_condition(">=", 5)