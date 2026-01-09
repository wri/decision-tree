import os
import yaml

from src.api_utils import get_ids, opentopo_pull_wrapper, get_tm_feats
from tests.tools import get_opentopo_api_key
from src.tools import get_project_root
import src.process_api_results as clean

ROOT_PATH = get_project_root()

parms_path = os.path.join(ROOT_PATH, "tests", "params.yaml")
with open(parms_path, 'r') as file:
    PARAMS = yaml.safe_load(file)


def test_tm_features():
    project_id = '468bee12-bfbc-4387-a00a-d7e915576427'
    features = _get_project_tm_features(project_id)

    # Confirm that the file contains an expected number of polygons or more
    expected_minimum_poly_count = 19
    assert len(features) >= expected_minimum_poly_count


def test_clean_tm_features():
    project_id = '468bee12-bfbc-4387-a00a-d7e915576427'
    features = _get_project_tm_features(project_id)

    cleaned_features = clean.process_tm_api_results(PARAMS, features)

    expected_minimum_poly_count = 19
    assert len(features) >= expected_minimum_poly_count

    actual_attribute_count = cleaned_features.shape[1]
    expected_column_count = 11
    assert actual_attribute_count == expected_column_count

    expected_columns = ['project_id', 'poly_id', 'site_id', 'project_name', 'geometry', 'plantstart', 'practice', 'target_sys', 'dist', 'project_phase', 'area']
    all_exist = all(col in cleaned_features.columns for col in expected_columns)
    assert all_exist


def test_slope_statistics(tmp_path):
    project_id = '44c352fb-0581-4b97-92d7-6cbe459e13d0'  # KEN_22_EFK
    features = _get_project_tm_features(project_id)
    cleaned_features = clean.process_tm_api_results(PARAMS, features)

    # get open_topo api key
    opentopo_key = get_opentopo_api_key(PARAMS)
    config = {"opentopo_key": opentopo_key}

    # Specify csv-path to temp directory
    sub_dir = os.path.join(tmp_path, "test_slope_stats.csv")
    PARAMS['outfile']['project_stats'] = sub_dir

    # TODO Why are there so many duplicates?
    slope_statistics = opentopo_pull_wrapper(PARAMS, config, cleaned_features)
    unique_stats = slope_statistics.drop_duplicates(subset=['project_id','poly_id'])

    assert len(unique_stats) == len(cleaned_features)

    actual_attribute_count = slope_statistics.shape[1]
    expected_attribute_count = 20
    assert actual_attribute_count == expected_attribute_count


def _get_project_tm_features(project_id):
    # Clear the tm_response outfile path
    PARAMS['outfile']['tm_response'] = ''
    PARAMS['outfile']['geojsons'] = os.path.join(ROOT_PATH, "tests", "data", "source_geojsons")

    features = get_tm_feats(PARAMS, [project_id])
    return features
