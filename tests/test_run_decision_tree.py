import os
from numpy import nan

from decision_tree.run_decision_tree import VerificationDecisionTree, main
from conftest import DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH, TEST_01_GRI_PROJECT_ID
from tools import has_expected_project_ev_values


def test_run_decision_tree_full():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_full.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    poly_results, prj_results = workflow.run_decision_tree(project_ids=None, limit_to_test_projects=True)

    # verify that a project was returned
    assert len(prj_results) >= 1


def test_run_decision_tree_projectids():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_projectids.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    project_ids = [TEST_01_GRI_PROJECT_ID]

    poly_results, prj_results = workflow.run_decision_tree(project_ids=project_ids, limit_to_test_projects=True)

    # REAL PROJECT BELOW - Only use for examination of an actual project
    # project_ids = [TEST_REAL_PROJECT_C1_ID]
    # poly_results, prj_results  = workflow.run_decision_tree(project_ids=project_ids)
    # REAL PROJECT ABOVE

    # TODO Determine how to modify the data to get more variation
    expected_poly_baseline_suitability = 13.3
    expected_poly_baseline_total = 0
    expected_poly_ev_total = 0

    expected_project_count = 1
    expected_project_ev_label= 'review required'
    expected_pct_area_scored = 0

    has_expected_project_ev_values(poly_results, prj_results,
                           expected_poly_baseline_suitability, expected_poly_baseline_total, expected_poly_ev_total,
                           expected_project_count, expected_project_ev_label, expected_pct_area_scored
                           )


def test_run_decision_tree_score():
    params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_score.yaml")

    workflow = main(params_path, SECRETS_FILE_PATH, parse_only=True)

    poly_results, prj_results = workflow.run_decision_tree()

    # TODO Determine how to modify the data to get more variation
    # verify that two projects were returned
    expected_poly_baseline_suitability = nan
    expected_poly_baseline_total = 0
    expected_poly_ev_total = 0

    expected_project_count = 1
    expected_project_ev_label= 'review required'
    expected_pct_area_scored = 0

    has_expected_project_ev_values(poly_results, prj_results,
                           expected_poly_baseline_suitability, expected_poly_baseline_total, expected_poly_ev_total,
                           expected_project_count, expected_project_ev_label, expected_pct_area_scored
                           )


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
