import os
import yaml

from decision_tree.api_utils import opentopo_pull_wrapper, get_geoparquet
from decision_tree.process_api_results import _read_geoparquet, flatten_tm_geoparquet
from decision_tree.process_api_results import process_tm_results
from decision_tree.tools import convert_to_os_path, load_secrets
from conftest import DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH
from tools import delete_source_geojsons_file

parms_path = os.path.join(DT_TEST_PARAMS_DIR, "params_full.yaml")
with open(parms_path, 'r') as file:
    PARAMS = yaml.safe_load(file)
SECRETS = load_secrets(SECRETS_FILE_PATH)


def test_tm_features():
    project_ids = ['468bee12-bfbc-4387-a00a-d7e915576427']
    _, features = _get_project_tm_features(project_ids)

    # cleanup
    delete_source_geojsons_file('468bee12-bfbc-4387-a00a-d7e915576427_07-14-2025.geojson')

    # Confirm that the file contains at least one polygon
    assert len(features) >= 1


def test_clean_tm_features():
    project_ids = ['468bee12-bfbc-4387-a00a-d7e915576427']
    parquet_outfile, features = _get_project_tm_features(project_ids)

    outfile = PARAMS['outfile']
    project_data_dir = outfile["project_data_folder"]
    GEOJSON_DIR = convert_to_os_path(project_data_dir, outfile['geojsons'])

    cleaned_features = process_tm_results(PARAMS, parquet_outfile, GEOJSON_DIR)

    # cleanup
    delete_source_geojsons_file('468bee12-bfbc-4387-a00a-d7e915576427_07-14-2025.geojson')

    assert len(cleaned_features) >= 1

    actual_attribute_count = cleaned_features.shape[1]
    expected_column_count = 12
    assert actual_attribute_count == expected_column_count

    expected_columns = ['cohort', 'project_id', 'poly_id', 'site_id', 'project_name', 'geometry', 'plantstart', 'practice', 'target_sys', 'dist', 'project_phase', 'area']
    all_exist = all(col in cleaned_features.columns for col in expected_columns)
    assert all_exist


def test_slope_statistics(tmp_path):
    project_ids = ['44c352fb-0581-4b97-92d7-6cbe459e13d0']  # KEN_22_EFK
    parquet_outfile, features = _get_project_tm_features(project_ids)

    outfile = PARAMS['outfile']
    project_data_dir = outfile["project_data_folder"]
    GEOJSON_DIR = convert_to_os_path(project_data_dir, outfile['geojsons'])

    cleaned_features = process_tm_results(PARAMS, parquet_outfile, GEOJSON_DIR)

    # Specify csv-path to temp directory
    sub_dir = os.path.join(tmp_path, "test_slope_stats.csv")
    PARAMS['outfile']['project_stats'] = sub_dir

    outfile = PARAMS['outfile']
    project_data_dir = outfile["project_data_folder"]
    geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])

    # Thin to the testing project
    cleaned_features = cleaned_features[cleaned_features['project_id'].isin(project_ids)].reset_index(drop=True)

    # get slope statistics
    slope_statistics = opentopo_pull_wrapper(PARAMS, SECRETS, geojson_dir, cleaned_features)

    # cleanup
    delete_source_geojsons_file('KEN_22_EFK_07-14-2025.geojson')

    assert len(slope_statistics) == len(cleaned_features)

    actual_attribute_count = slope_statistics.shape[1]
    expected_attribute_count = 19
    assert actual_attribute_count == expected_attribute_count


def _get_project_tm_features(project_ids):
    outfile = PARAMS['outfile']
    data_v = outfile["data_version"]
    project_data_dir = outfile["project_data_folder"]
    parquet_outfile = convert_to_os_path(project_data_dir, outfile['geoparquet'].format(cohort=outfile['cohort'], data_version=data_v))

    features = get_geoparquet(PARAMS, SECRETS, parquet_outfile)
    df = _read_geoparquet(features)
    raw_df = flatten_tm_geoparquet(df)

    # Thin to projects
    features = raw_df[raw_df['project_id'].isin(project_ids)].reset_index(drop=True)

    return parquet_outfile, features

