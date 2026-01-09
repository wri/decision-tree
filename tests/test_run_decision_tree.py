import math
import os
import pandas as pd
import csv
from pathlib import Path
import yaml

from run_decision_tree import main, VerificationDecisionTree, compute_project_results, compute_ev_statistics
from src.api_utils import opentopo_pull_wrapper
from tests.tools import get_project_root, get_opentopo_api_key, standardize_test_param_paths, delete_scratch_file, \
    delete_source_geojsons_file

ROOT_PATH = get_project_root()
PARAMS_PATH = os.path.join(ROOT_PATH, "tests", "params.yaml")
SECRETS_PATH = os.path.join(ROOT_PATH, "secrets.yaml")
with open(PARAMS_PATH, 'r') as file:
    params = yaml.safe_load(file)
PARAMS = standardize_test_param_paths(params)

PROJECT_NAME_A = 'TGO_22_APAF'
PROJECT_NAME_B = 'RWA_22_ICRAF'


def test_run_decision_tree_full():
    project_ids_path = os.path.join(ROOT_PATH, "tests", "data", "source_csv", "prj_ids_c2_10-06-25.csv")
    # project_ids_path = os.path.join(ROOT_PATH, "data", "portfolio_csvs", "prj_ids_c2_10-06-25.csv")

    project_ids = [value for row in (list(csv.reader(open(project_ids_path)))[1:]) for value in row]

    workflow = main(PARAMS_PATH, SECRETS_PATH, save_to_asana=False, parse_only=True)

    slope_statistics, prj_results = workflow.run(project_ids, save_to_asana = False)

    # cleanup
    delete_source_geojsons_file('GHA_22_TILAA_07-14-2025.geojson')
    delete_source_geojsons_file('TAZ_22_KIJANIP_07-14-2025.geojson')

    expected_project_count = 2
    assert len(prj_results) == expected_project_count

    sample_project_id = 'd1f355a2-3e0f-4ffd-bb2f-eec104bf8442'
    expected_sum_median_slope = 481.5
    actual_sum_median_slope = slope_statistics[slope_statistics['project_id'] == sample_project_id]['median_slope'].sum()
    assert math.isclose(actual_sum_median_slope, expected_sum_median_slope, rel_tol=0.1), f"Expected {expected_sum_median_slope}, got {actual_sum_median_slope}"

def test_run_decision_tree_partial():
    sample_tm_file = "tm_api_c1_07-14-2025.csv"
    sample_maxar_file = "comb_img_availability_c1_07-14-2025.csv"
    sample_tm_data_path = os.path.join(ROOT_PATH, "tests", "data", "source_csv", sample_tm_file)
    sample_maxar_data_path = os.path.join(ROOT_PATH, "tests", "data", "source_csv", sample_maxar_file)

    # Setup for testing
    opentopo_key = get_opentopo_api_key(PARAMS)
    opentopo_secrets = {"opentopo_key": opentopo_key}
    # Scratch folder
    out_directory = Path(PARAMS['outfile']['project_stats']).parent
    os.makedirs(out_directory, exist_ok=True)

    # read sample data
    tm_clean = pd.read_csv(sample_tm_data_path)

    # compute slope stats
    slope_statistics = opentopo_pull_wrapper(PARAMS, opentopo_secrets, tm_clean)

    # compute ev decision
    ev = compute_ev_statistics(PARAMS, tm_clean, sample_maxar_data_path, slope_statistics)

    # Clean up scratch file
    delete_scratch_file('TGO_22_APAF_slope_stats.csv')
    delete_scratch_file('RWA_22_ICRAF_slope_stats.csv')

    # verify that two projects were returned
    project_count = ev['project_name'].nunique()
    assert project_count == 2

    # for project_b verify counts for a representative ev_decision
    unique_counts = ev.groupby('project_name')['ev_decision'].nunique().reset_index()
    proj_a_decision_variant_count = unique_counts.loc[unique_counts['project_name'] == PROJECT_NAME_B]['ev_decision'].iloc[0]
    expected_proj_a_decision_variant_count = 4
    assert proj_a_decision_variant_count == expected_proj_a_decision_variant_count

    # for project_b verify ev_decision for a specific polygon
    proj_a_poly_decision =  ev[(ev['project_name'] == PROJECT_NAME_A) & (ev['poly_id'] == 'b0b8236e-b4e2-48b3-a3a2-d1b90b3cf058')]['ev_decision'].iloc[0]
    expected_proj_a_poly_decision = 'strong field'
    assert proj_a_poly_decision == expected_proj_a_poly_decision

def test_run_decision_tree_score():
    sample_file = "dtree_output_c1_07-14-2025_exp5.csv"
    sample_data_path = os.path.join(ROOT_PATH, "tests", "data", "source_csv", sample_file)
    ev = pd.read_csv(sample_data_path)

    poly_results, prj_results = compute_project_results(PARAMS, ev)

    # verify that two projects were returned
    project_count = prj_results['project_name'].nunique()
    assert project_count == 2

    # for project__b, evaluate ev-cost
    project_e_costs = poly_results.groupby('project_name')['ev_cost'].sum().reset_index()
    proj_a_ev_cost = project_e_costs.loc[project_e_costs['project_name'] == PROJECT_NAME_B, 'ev_cost'].iloc[0]
    expected_proj_a_ev_cost = 300.4
    assert math.isclose(proj_a_ev_cost, expected_proj_a_ev_cost, rel_tol=0, abs_tol=0.0),\
        f"proj: {PROJECT_NAME_B}: actual_ev_cost:{proj_a_ev_cost} != expected_cost:{expected_proj_a_ev_cost}"

    #  for project__b, evaluate baseline-cost
    project_baseline_costs = poly_results.groupby('project_name')['baseline_cost'].sum().reset_index()
    proj_a_baseline_cost = project_baseline_costs.loc[project_baseline_costs['project_name'] == PROJECT_NAME_B, 'baseline_cost'].iloc[0]
    expected_proj_a_baseline_cost = 1078.6
    assert math.isclose(proj_a_baseline_cost, expected_proj_a_baseline_cost, rel_tol=0, abs_tol=0.0), \
        f"proj: {PROJECT_NAME_B}: actual_baseline_cost:{proj_a_baseline_cost} != expected_cost:{expected_proj_a_baseline_cost}"

    # for project__b, evaluate baseline_project_score_0_100
    proj_a_proj_baseline_score= prj_results.loc[prj_results['project_name'] == PROJECT_NAME_B, 'baseline_project_score_0_100'].iloc[0]
    expected_proj_a_baseline_score = 82.8
    assert math.isclose(proj_a_proj_baseline_score, expected_proj_a_baseline_score, rel_tol=0, abs_tol=0.0), \
        f"proj: {PROJECT_NAME_B}: actual_baseline_score:{proj_a_baseline_cost} != expected_score:{expected_proj_a_baseline_cost}"


def test_run_decision_tree_param_parsing():
    workflow = main(PARAMS_PATH, SECRETS_PATH, save_to_asana=False, parse_only=True)

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
