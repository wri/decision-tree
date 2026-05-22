import os

import yaml
from gri_shared_library.os_tools import create_folder

from conftest import DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH, TEST_01_GRI_PROJECT_ID, TEST_REAL_PROJECT_C1_ID
from decision_tree.api_utils import opentopo_pull_wrapper, get_geoparquet
from decision_tree.process_api_results import _read_geoparquet, flatten_tm_geoparquet
from decision_tree.process_api_results import process_tm_results
from decision_tree.tools import convert_to_os_path, load_secrets

params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_full.yaml")
with open(params_path, 'r') as file:
    PARAMS = yaml.safe_load(file)
SECRETS = load_secrets(SECRETS_FILE_PATH)


def test_tm_features():
    project_ids = [TEST_01_GRI_PROJECT_ID]
    parquet_outfile, features = _get_project_tm_features(project_ids)

    # Confirm that the file contains at least one polygon
    assert len(features) == 3


def test_clean_tm_features():
    project_ids = [TEST_01_GRI_PROJECT_ID]
    parquet_outfile, features = _get_project_tm_features(project_ids)

    outfile = PARAMS['outfile']
    project_data_dir = outfile["project_data_folder"]
    geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])

    cleaned_features = process_tm_results(params=PARAMS, results=parquet_outfile, geojson_dir=geojson_dir,
                                          project_ids=project_ids, limit_to_test_projects=True)

    assert len(cleaned_features) == 3

    actual_attribute_count = cleaned_features.shape[1]
    expected_column_count = 11
    assert actual_attribute_count == expected_column_count

    expected_columns = ['cohort', 'project_id', 'poly_id', 'site_id', 'project_name', 'geometry', 'plantstart', 'practice', 'target_sys', 'area', 'notes']
    all_exist = all(col in cleaned_features.columns for col in expected_columns)
    assert all_exist


def test_slope_statistics(tmp_path):
    project_ids = [TEST_01_GRI_PROJECT_ID]; limit_to_test_projects = True
    # REAL PROJECT BELOW - Only use for examination of an actual project
    project_ids = [TEST_REAL_PROJECT_C1_ID]; limit_to_test_projects = False
    # slope_statistics, poly_results, prj_results = workflow.run_decision_tree(project_ids=project_ids)
    # REAL PROJECT ABOVE

    parquet_outfile, features = _get_project_tm_features(project_ids)

    # Specify csv-path to temp directory
    sub_dir = os.path.join(tmp_path, "test_slope_stats.csv")
    PARAMS['outfile']['project_stats'] = sub_dir

    outfile = PARAMS['outfile']
    project_data_dir = outfile["project_data_folder"]
    geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])

    cleaned_features = process_tm_results(params=PARAMS, results=parquet_outfile, geojson_dir=geojson_dir,
                                          project_ids=project_ids, limit_to_test_projects=limit_to_test_projects)

    # get slope statistics
    slope_statistics = opentopo_pull_wrapper(PARAMS, SECRETS, geojson_dir, cleaned_features)

    assert len(slope_statistics) == len(cleaned_features)

    actual_attribute_count = slope_statistics.shape[1]
    expected_attribute_count = 18
    assert actual_attribute_count == expected_attribute_count


def _get_project_tm_features(project_ids):
    outfile = PARAMS['outfile']
    data_v = outfile["data_version"]
    project_data_dir = outfile["project_data_folder"]
    parquet_outfile = convert_to_os_path(project_data_dir, outfile['geoparquet'].format(cohort=outfile['cohort'], data_version=data_v))

    # cleanup target
    tm_raw_dir = os.path.dirname(parquet_outfile)
    create_folder(tm_raw_dir)

    features = get_geoparquet(PARAMS, SECRETS, parquet_outfile)
    df = _read_geoparquet(features)
    raw_df = flatten_tm_geoparquet(df)

    # Thin to projects
    features = raw_df[raw_df['project_id'].isin(project_ids)].reset_index(drop=True)

    return parquet_outfile, features

