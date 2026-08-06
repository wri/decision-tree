import os

from numpy import nan

from conftest import DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH
from decision_tree.constants import TestProjectHandling
from decision_tree.run_decision_tree import VerificationDecisionTree, main
from tools import has_expected_project_ev_values, folder_cleanup
from gri_shared_library.constants_for_tests import TEST_01_GRI_PROJECT_ID, TEST_REAL_PROJECT_C1_ID


def test_run_decision_tree_full():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_full.yaml")
    # pre-run cleanup
    folder_cleanup(params_path)

    test_project_handling = TestProjectHandling.ONLY
    try:
        workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)
        poly_results, prj_results = workflow.run_decision_tree(project_ids=None, test_project_handling=test_project_handling)

        # verify that a project was returned
        assert len(prj_results) >= 1

    finally:
        # post-run cleanup
        folder_cleanup(params_path)


def test_run_decision_tree_projectids():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_projectids.yaml")
    # pre-run cleanup
    folder_cleanup(params_path)

    project_ids = [TEST_01_GRI_PROJECT_ID]; test_project_handling = TestProjectHandling.ONLY
    # REAL PROJECT BELOW - Only use for examination of an actual project
    # project_ids = [TEST_REAL_PROJECT_C1_ID]; test_project_handling = TestProjectHandling.EXCLUDE
    # REAL PROJECT ABOVE

    try:
        workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)
        poly_results, prj_results = workflow.run_decision_tree(project_ids=project_ids,
                                                               test_project_handling=test_project_handling)

        # TODO Determine how to modify the data to get more variation
        if project_ids == [TEST_01_GRI_PROJECT_ID]:
            sample_poly_id = 'a09e8e9f-f0f6-438a-95ac-aa5457082d46'
            expected_poly_baseline_suitability = 60.0
            expected_poly_baseline_total = 61.2
            expected_poly_ev_total = 61.2

            expected_project_count = 1
            expected_project_ev_label= 'weak field'
            expected_pct_area_scored = 1.0
        else:
            sample_poly_id = 'b25a7139-6597-44ab-9765-b626745c97bc'
            expected_poly_baseline_suitability = 21.1
            expected_poly_baseline_total = 3033.6
            expected_poly_ev_total = 964.4
            expected_project_count = 1
            expected_project_ev_label = 'not available'
            expected_pct_area_scored = 0.318

        has_expected_project_ev_values(poly_results, prj_results, sample_poly_id,
                               expected_poly_baseline_suitability, expected_poly_baseline_total, expected_poly_ev_total,
                               expected_project_count, expected_project_ev_label, expected_pct_area_scored
                               )

    finally:
        # post-run cleanup
        folder_cleanup(params_path)


def test_run_decision_tree_projectids_api_query():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_projectids_api.yaml")
    # pre-run cleanup
    folder_cleanup(params_path)

    project_ids = [TEST_REAL_PROJECT_C1_ID]; test_project_handling = TestProjectHandling.EXCLUDE
    # project_ids = ["562fa859-5124-49a5-947f-e1ddb7680e07"]; test_project_handling = TestProjectHandling.EXCLUDE # KEN_23_WWF
    # project_ids = ["15a3e7a4-5ebd-45d0-a072-d78d2b5f7ac6"]; test_project_handling = TestProjectHandling.EXCLUDE  # KEN_23_ILEG
    # project_ids = ["a727d737-370f-4ac0-889c-88b7b190e4ff"]; test_project_handling = TestProjectHandling.EXCLUDE # GHA_23_EAF

    try:
        workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)
        poly_results, prj_results = (
            workflow.run_decision_tree(project_ids=project_ids, test_project_handling=test_project_handling))

        # expected results
        if project_ids == [TEST_REAL_PROJECT_C1_ID]:
            sample_poly_id = 'b25a7139-6597-44ab-9765-b626745c97bc'
            expected_poly_baseline_suitability = 21.1
            expected_poly_baseline_total = 3033.6
            expected_poly_ev_total = 964.4
            expected_project_count = 1
            expected_project_ev_label= 'not available'
            expected_pct_area_scored = 0.318
        else:
            assert False # not a recognized project

        has_expected_project_ev_values(poly_results, prj_results, sample_poly_id,
                               expected_poly_baseline_suitability, expected_poly_baseline_total, expected_poly_ev_total,
                               expected_project_count, expected_project_ev_label, expected_pct_area_scored
                               )

    finally:
        # post-run cleanup
        folder_cleanup(params_path)


def test_run_decision_tree_score():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_score.yaml")
    # pre-run cleanup
    folder_cleanup(params_path)

    try:
        workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)
        poly_results, prj_results = workflow.run_decision_tree()

        # TODO Determine how to modify the data to get more variation
        # verify that two projects were returned
        sample_poly_id = 'a09e8e9f-f0f6-438a-95ac-aa5457082d46'
        expected_poly_baseline_suitability = nan
        expected_poly_baseline_total = 0
        expected_poly_ev_total = 0

        expected_project_count = 1
        expected_project_ev_label= 'review required'
        expected_pct_area_scored = 0

        has_expected_project_ev_values(poly_results, prj_results, sample_poly_id,
                               expected_poly_baseline_suitability, expected_poly_baseline_total, expected_poly_ev_total,
                               expected_project_count, expected_project_ev_label, expected_pct_area_scored
                               )

    finally:
        # post-run cleanup
        folder_cleanup(params_path)


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


