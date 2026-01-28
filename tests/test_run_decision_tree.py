import os

from decision_tree_src.run_decision_tree import VerificationDecisionTree
from decision_tree_src.tools import get_project_root
from runner import main

ROOT_PATH = get_project_root()
PARAMS_DIR = os.path.join(ROOT_PATH, "tests")
SECRETS_PATH = os.path.join(ROOT_PATH, "secrets.yaml")

# TODO - Three tests are commented out pending creation of imaginary project data by John
# def test_run_decision_tree_full():
#     project_ids_path = os.path.join(ROOT_PATH, "tests", "data", "source_csv", "prj_ids_c2_10-06-25.csv")
#     # project_ids_path = os.path.join(ROOT_PATH, "data", "portfolio_csvs", "prj_ids_c2_10-06-25.csv")
#
#     project_ids = [value for row in (list(csv.reader(open(project_ids_path)))[1:]) for value in row]
#
#     params_path = os.path.join(PARAMS_DIR, "params_full.yaml")
#     workflow = main(params_path, SECRETS_PATH, parse_only=True)
#
#     slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids)
#
#     # cleanup
#     delete_source_geojsons_file('GHA_22_TILAA_07-14-2025.geojson')
#     delete_source_geojsons_file('TAZ_22_KIJANIP_07-14-2025.geojson')
#     delete_scratch_file('GHA_22_TILAA_slope_stats.csv')
#     delete_scratch_file('TAZ_22_KIJANIP_slope_stats.csv')
#
#     expected_project_count = 2
#     assert len(prj_results) == expected_project_count
#
#     sample_project_id = 'd1f355a2-3e0f-4ffd-bb2f-eec104bf8442'
#     expected_sum_median_slope = 481.5
#     actual_sum_median_slope = slope_statistics[slope_statistics['project_id'] == sample_project_id]['median_slope'].sum()
#     assert math.isclose(actual_sum_median_slope, expected_sum_median_slope, rel_tol=0.1), f"Expected {expected_sum_median_slope}, got {actual_sum_median_slope}"
#
#     expected_project_label = 'review required'
#     actual_project_label = prj_results[prj_results['project_id'] == sample_project_id]['baseline_project_label'].values[0]
#     assert actual_project_label == expected_project_label, f"Expected: {expected_project_label!r}, Actual: {actual_project_label!r}"
#
#
# def test_run_decision_tree_partial():
#     params_path = os.path.join(PARAMS_DIR, "params_partial.yaml")
#     workflow = main(params_path, SECRETS_PATH, parse_only=True)
#
#     slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)
#
#     # Clean up scratch file
#     delete_scratch_file('RWA_22_ICRAF_slope_stats.csv')
#     delete_scratch_file('TGO_22_APAF_slope_stats.csv')
#
#     # verify that two projects were returned
#     expected_project_count = 2
#     assert len(prj_results) == expected_project_count
#
#     sample_project_id = 'f81c1422-025c-45b1-a2e1-d354177523ca'
#     expected_sum_median_slope = 307.4
#     actual_sum_median_slope = slope_statistics[slope_statistics['project_id'] == sample_project_id]['median_slope'].sum()
#     assert math.isclose(actual_sum_median_slope, expected_sum_median_slope, rel_tol=0.1), f"Expected {expected_sum_median_slope}, got {actual_sum_median_slope}"
#
#     expected_project_label = 'strong remote'
#     actual_project_label = prj_results[prj_results['project_id'] == sample_project_id]['baseline_project_label'].values[0]
#     assert actual_project_label == expected_project_label, f"Expected: {expected_project_label!r}, Actual: {actual_project_label!r}"


def test_run_decision_tree_score():
    params_path = os.path.join(PARAMS_DIR, "params_score.yaml")
    workflow = main(params_path, SECRETS_PATH, parse_only=True)

    slope_statistics, poly_results, prj_results = workflow.run_decision_tree(None)

    # verify that two projects were returned
    expected_project_count = 2
    assert len(prj_results) == expected_project_count

    sample_project_id = 'f81c1422-025c-45b1-a2e1-d354177523ca'
    expected_project_label = 'strong remote'
    actual_project_label = prj_results[prj_results['project_id'] == sample_project_id]['baseline_project_label'].values[0]
    assert actual_project_label == expected_project_label, f"Expected: {expected_project_label!r}, Actual: {actual_project_label!r}"


def test_run_decision_tree_param_parsing():
    params_path = os.path.join(PARAMS_DIR, "params_full.yaml")
    workflow = main(params_path, SECRETS_PATH, parse_only=True)

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
