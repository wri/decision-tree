import math
import os
import pytest

from decision_tree.run_decision_tree import VerificationDecisionTree
from decision_tree.tools import get_project_root
from run_app import main
from tests.tools import delete_file, delete_folder

ROOT_PATH = get_project_root()
SECRETS_PATH = os.path.join(ROOT_PATH, "secrets.yaml")
TEST_DIR = os.path.join(ROOT_PATH, "tests")
TEST_SECRETS_PATH = os.path.join(TEST_DIR, "secrets.yaml")


@pytest.mark.skip(reason="This option is very slow and expensive so not normally executed.")
def test_run_decision_tree_full():
    test_project = os.path.join(TEST_DIR, "data", "test_projects", "test_01_gri")
    params_path = os.path.join(test_project, "params_full.yaml")

    workflow = main(params_path, SECRETS_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that a project was returned
    assert len(prj_results) >= 1

    # Clean up temporary folders
    delete_folder(os.path.join(test_project, "slope"))
    delete_folder(os.path.join(test_project, "tm_api_response"))


def test_run_decision_tree_id_list():
    test_project = os.path.join(TEST_DIR, "data", "test_projects", "test_01_gri")
    params_path = os.path.join(test_project, "params_id_list.yaml")

    workflow = main(params_path, SECRETS_PATH, parse_only=True)

    project_ids = ['1826cc5f-0d4d-4427-b5b3-fe244deba919'] # TEST_01_GRI
    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids)

    expected_project_count = 1
    expected_polygon_count= 3
    expected_median_slope = 34.4
    expected_project_label = 'review required'
    _has_expected_values(slope_statistics, prj_results, expected_project_count, expected_polygon_count,
                         expected_project_label, expected_median_slope)

    # Clean up temporary folders
    delete_folder(os.path.join(test_project, "slope"))
    delete_folder(os.path.join(test_project, "tm_api_response"))


def test_run_decision_tree_partial():
    test_project = os.path.join(TEST_DIR, "data", "test_projects", "test_01_gri")
    params_path = os.path.join(test_project, "params_partial.yaml")

    workflow = main(params_path, SECRETS_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that two projects were returned
    expected_project_count = 1
    expected_polygon_count= 3
    expected_project_label = 'review required'
    expected_median_slope = 34.4
    _has_expected_values(slope_statistics, prj_results, expected_project_count, expected_polygon_count,
                         expected_project_label, expected_median_slope)

    # Clean up scratch folders
    delete_folder(os.path.join(test_project, "slope"))


def test_run_decision_tree_score():
    test_project = os.path.join(TEST_DIR, "data", "test_projects", "test_01_gri")
    params_path = os.path.join(test_project, "params_score.yaml")

    workflow = main(params_path, SECRETS_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that two projects were returned
    expected_project_count = 1
    expected_polygon_count= 3
    expected_project_label = 'review required'
    expected_median_slope = None
    _has_expected_values(slope_statistics, prj_results, expected_project_count, expected_polygon_count,
                         expected_project_label, expected_median_slope)


def test_run_decision_tree_param_parsing():
    params_path = os.path.join(TEST_DIR, "params_for_parse_test.yaml")
    workflow = main(params_path, TEST_SECRETS_PATH, parse_only=True)

    expected_atts = {"params", "portfolio", "tm_outfile", "slope_stats", "project_feats", "project_feats_maxar",
                     "maxar_meta", "tree_results", "poly_score", "prj_score"}

    assert isinstance(workflow, VerificationDecisionTree)
    assert _has_expected_attributes(workflow, expected_atts)


def _has_expected_values(slope_statistics, prj_results, expected_project_count, expected_polygon_count,
                         expected_project_label, expected_median_slope):
    assert len(prj_results) == expected_project_count

    assert prj_results['ev_total_poly_count'].values[0] == expected_polygon_count

    actual_project_label = prj_results['baseline_project_label'].values[0]
    assert actual_project_label == expected_project_label, f"Expected: {expected_project_label!r}, Actual: {actual_project_label!r}"

    if expected_median_slope is not None:
        actual_median_slope = slope_statistics['median_slope'].median()
        assert math.isclose(actual_median_slope, expected_median_slope, rel_tol=0.1), f"Expected {expected_median_slope}, got {actual_median_slope}"
    else:
        assert slope_statistics is None, f"Expected slop_statistics to be None, but got values."


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
