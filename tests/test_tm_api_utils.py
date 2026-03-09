import os
import yaml

from decision_tree.api_utils import opentopo_pull_wrapper, get_tm_feats
from decision_tree.process_api_results import process_tm_api_results
from decision_tree.tools import get_project_root, convert_to_os_path, load_secrets
from tests.tools import get_opentopo_api_key, delete_source_geojsons_file

ROOT_PATH = get_project_root()

parms_path = os.path.join(ROOT_PATH, "tests", "param_files", "params_full.yaml")
with open(parms_path, 'r') as file:
    PARAMS = yaml.safe_load(file)
SECRETS = load_secrets(os.path.join(ROOT_PATH, "secrets.yaml"))


def test_tm_features():
    project_id = '468bee12-bfbc-4387-a00a-d7e915576427'
    features = _get_project_tm_features(project_id)

    # cleanup
    delete_source_geojsons_file('468bee12-bfbc-4387-a00a-d7e915576427_07-14-2025.geojson')

    # Confirm that the file contains at least one polygon
    assert len(features) >= 1


def test_clean_tm_features():
    project_id = '468bee12-bfbc-4387-a00a-d7e915576427'
    features = _get_project_tm_features(project_id)

    cleaned_features = process_tm_api_results(PARAMS, features)

    # cleanup
    delete_source_geojsons_file('468bee12-bfbc-4387-a00a-d7e915576427_07-14-2025.geojson')

    assert len(features) >= 1

    actual_attribute_count = cleaned_features.shape[1]
    expected_column_count = 11
    assert actual_attribute_count == expected_column_count

    expected_columns = ['project_id', 'poly_id', 'site_id', 'project_name', 'geometry', 'plantstart', 'practice', 'target_sys', 'dist', 'project_phase', 'area']
    all_exist = all(col in cleaned_features.columns for col in expected_columns)
    assert all_exist


def test_slope_statistics(tmp_path):
    project_id = '44c352fb-0581-4b97-92d7-6cbe459e13d0'  # KEN_22_EFK
    features = _get_project_tm_features(project_id)
    cleaned_features = process_tm_api_results(PARAMS, features)

    # Specify csv-path to temp directory
    sub_dir = os.path.join(tmp_path, "test_slope_stats.csv")
    PARAMS['outfile']['project_stats'] = sub_dir

    outfile = PARAMS['outfile']
    project_data_dir = outfile["project_data_folder"]
    geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])

    slope_statistics = opentopo_pull_wrapper(PARAMS, SECRETS, geojson_dir, cleaned_features)

    # cleanup
    delete_source_geojsons_file('KEN_22_EFK_07-14-2025.geojson')

    assert len(slope_statistics) == len(cleaned_features)

    actual_attribute_count = slope_statistics.shape[1]
    expected_attribute_count = 20
    assert actual_attribute_count == expected_attribute_count


def _get_project_tm_features(project_id):
    outfile = PARAMS['outfile']
    data_v = outfile["data_version"]
    project_data_dir = outfile["project_data_folder"]

    geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])
    tm_outfile = convert_to_os_path(project_data_dir, outfile['tm_response'].format(cohort=outfile['cohort'], data_version=data_v))

    features = get_tm_feats(PARAMS, SECRETS, geojson_dir, tm_outfile, [project_id])
    return features
