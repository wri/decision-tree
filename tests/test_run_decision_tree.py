import os
import pytest
from gri_shared_library.os_tools import remove_folder

from decision_tree.run_decision_tree import VerificationDecisionTree, main
from conftest import DT_TEST_PROJECTS, DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH, TEST_01_GRI_PROJECT_ID
from tools import has_expected_project_ev_values


@pytest.mark.skip(reason="This option is very slow and expensive so not normally executed.")
def test_run_decision_tree_full():
    test_project = os.path.join(DT_TEST_PROJECTS, "test_01_gri")
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_full.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that a project was returned
    assert len(prj_results) >= 1

    # Clean up temporary folders
    remove_folder(os.path.join(test_project, "slope"))
    remove_folder(os.path.join(test_project, "tm_api_response"))


def test_run_decision_tree_projectids():
    test_project = os.path.join(DT_TEST_PROJECTS, "test_01_gri")
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_projectids.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    project_ids = [TEST_01_GRI_PROJECT_ID] # TEST_01_GRI
    # project_ids = ['d1f355a2-3e0f-4ffd-bb2f-eec104bf8442'] # Only use for examination of an actual project
    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids)

    expected_project_count = 1
    expected_polygon_count= 3
    expected_median_slope = 34.4
    expected_project_label = 'review required'
    expected_baseline_total = 0 # TODO Determine how to modify the data to get more variation
    expected_ev_total = 0 # TODO Determine how to modify the data to get more variation
    has_expected_project_ev_values(slope_statistics, poly_results, prj_results, expected_project_count, expected_polygon_count,
                                   expected_project_label, expected_median_slope, expected_baseline_total, expected_ev_total)

    # Clean up temporary folders
    remove_folder(os.path.join(test_project, "slope"))
    remove_folder(os.path.join(test_project, "tm_api_response"))


def test_run_decision_tree_score():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_score.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

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
