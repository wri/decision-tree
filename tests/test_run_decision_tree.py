import os
import pytest
import yaml

from decision_tree.run_decision_tree import VerificationDecisionTree, main
from conftest import DT_TEST_PROJECTS, DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH, TEST_01_GRI_PROJECT_ID, \
    TEST_REAL_PROJECT_C1_ID
from tools import has_expected_project_ev_values


def test_run_decision_tree_full():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_full.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids=None, limit_to_test_projects=True)

    # verify that a project was returned
    assert len(prj_results) >= 1


def test_run_decision_tree_projectids():
    test_project = os.path.join(DT_TEST_PROJECTS, "test_01_gri")
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_projectids.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    project_ids = [TEST_01_GRI_PROJECT_ID]

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids=project_ids, limit_to_test_projects=True)

    # REAL PROJECT BELOW - Only use for examination of an actual project
    # project_ids = [TEST_REAL_PROJECT_C1_ID]
    # slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids=project_ids)
    # REAL PROJECT ABOVE

    expected_project_count = 1
    expected_polygon_count= 3
    expected_median_slope = 34.4
    expected_project_label = 'not available' # 'review required'
    expected_baseline_total = 0 # TODO Determine how to modify the data to get more variation
    expected_ev_total = 0 # TODO Determine how to modify the data to get more variation
    has_expected_project_ev_values(slope_statistics, poly_results, prj_results, expected_project_count, expected_polygon_count,
                                   expected_project_label, expected_median_slope, expected_baseline_total, expected_ev_total)


def test_run_decision_tree_score():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_score.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree()

    # verify that two projects were returned
    expected_project_count = 1
    expected_polygon_count= 3
    expected_project_label = 'review required'
    expected_median_slope = None
    expected_baseline_total = 0 # TODO Determine how to modify the data to get more variation
    expected_ev_total = 0 # TODO Determine how to modify the data to get more variation
    has_expected_project_ev_values(slope_statistics, poly_results, prj_results, expected_project_count, expected_polygon_count,
                                   expected_project_label, expected_median_slope, expected_baseline_total, expected_ev_total)


def test_run_decision_tree_param_parsing():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_for_parse_test.yaml")
    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    expected_atts = {"params", "checkpoint", "feats", "geojson_dir", "maxar_meta", "mode",  "poly_score", "portfolio",
                     "prj_score", "project_feats_maxar", "rules", "secrets", "slope_stats", "tm_raw", "tree_results"}

    assert isinstance(workflow, VerificationDecisionTree)
    assert _has_expected_attributes(workflow, expected_atts)


# Function to check if a class or object has all expected attributes
def _has_expected_attributes(obj, expected_attrs):
    """
    Check if the given object or class has all attributes in expected_attrs.

    Args:
        obj: The class or instance to check.
        expected_attrs (iterable): List or set of attribute names (strings).

    Returns:
        bool: True if all attributes exist, False otherwise.
    """
    if not hasattr(expected_attrs, '__iter__'):
        raise TypeError("expected_attrs must be an iterable of strings")
    if not all(isinstance(attr, str) for attr in expected_attrs):
        raise ValueError("All attribute names must be strings")

    return all(hasattr(obj, attr) for attr in expected_attrs)
