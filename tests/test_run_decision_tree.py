import os
import pytest

from decision_tree.run_decision_tree import VerificationDecisionTree
from decision_tree.tools import get_project_root
from run_app import main
from tests.tools import delete_folder, has_expected_project_ev_values

ROOT_PATH = get_project_root()
SECRETS_FILE_PATH = os.path.join(ROOT_PATH, "secrets.yaml")
TEST_PROJECTS = os.path.join(ROOT_PATH, "tests", "data", "test_projects")
TEST_PARAMS_DIR = os.path.join(ROOT_PATH, "tests", "param_files")

TEST_01_GRI_PROJECT_ID = '1826cc5f-0d4d-4427-b5b3-fe244deba919'


@pytest.mark.skip(reason="This option is very slow and expensive so not normally executed.")
def test_run_decision_tree_full():
    test_project = os.path.join(TEST_PROJECTS, "test_01_gri")
    params_path = os.path.join(TEST_PARAMS_DIR, "params_full.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that a project was returned
    assert len(prj_results) >= 1

    # Clean up temporary folders
    delete_folder(os.path.join(test_project, "slope"))
    delete_folder(os.path.join(test_project, "tm_api_response"))


def test_run_decision_tree_id_list():
    test_project = os.path.join(TEST_PROJECTS, "test_01_gri")
    params_path = os.path.join(TEST_PARAMS_DIR, "params_id_list.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    project_ids = [TEST_01_GRI_PROJECT_ID] # TEST_01_GRI
    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids)

    expected_project_count = 1
    expected_polygon_count= 3
    expected_median_slope = 34.4
    expected_project_label = 'review required'
    has_expected_project_ev_values(slope_statistics, prj_results, expected_project_count, expected_polygon_count,
                                   expected_project_label, expected_median_slope)

    # Clean up temporary folders
    delete_folder(os.path.join(test_project, "slope"))
    delete_folder(os.path.join(test_project, "tm_api_response"))


def test_run_decision_tree_partial():
    test_project = os.path.join(TEST_PROJECTS, "test_01_gri")
    params_path = os.path.join(TEST_PARAMS_DIR, "params_partial.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that two projects were returned
    expected_project_count = 1
    expected_polygon_count= 3
    expected_project_label = 'review required'
    expected_median_slope = 34.4
    has_expected_project_ev_values(slope_statistics, prj_results, expected_project_count, expected_polygon_count,
                                   expected_project_label, expected_median_slope)

    # Clean up scratch folders
    delete_folder(os.path.join(test_project, "slope"))


def test_run_decision_tree_score():
    params_path = os.path.join(TEST_PARAMS_DIR, "params_score.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that two projects were returned
    expected_project_count = 1
    expected_polygon_count= 3
    expected_project_label = 'review required'
    expected_median_slope = None
    has_expected_project_ev_values(slope_statistics, prj_results, expected_project_count, expected_polygon_count,
                                   expected_project_label, expected_median_slope)


def test_run_decision_tree_param_parsing():
    params_path = os.path.join(TEST_PARAMS_DIR, "params_for_parse_test.yaml")
    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    expected_atts = {"params", "portfolio", "tm_outfile", "slope_stats", "project_feats", "project_feats_maxar",
                     "maxar_meta", "tree_results", "poly_score", "prj_score"}

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
