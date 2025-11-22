import os
import yaml

from src.api_utils import get_ids, tm_pull_wrapper
from tests.tools import get_project_root
import src.process_api_results as clean

ROOT_PATH = get_project_root()

parms_path = os.path.join(ROOT_PATH, "tests", "data", "params.yaml")
with open(parms_path, 'r') as file:
    PARAMS = yaml.safe_load(file)

# TODO Too slow to test
# def test_get_ids():
#     ids = get_ids(PARAMS)


def test_tm_features():
    project_id = '468bee12-bfbc-4387-a00a-d7e915576427'
    features = get_project_tm_features(project_id)

    # Confirm that the file contains an expected number of polygons or more
    expected_minimum_poly_count = 24
    assert len(features) >= expected_minimum_poly_count


def test_clean_tm_features():
    project_id = '468bee12-bfbc-4387-a00a-d7e915576427'
    features = get_project_tm_features(project_id)

    cleaned_features = clean.process_tm_api_results(PARAMS, features)

    expected_minimum_poly_count = 24
    assert len(features) >= expected_minimum_poly_count

    expected_column_count = 13
    assert cleaned_features.shape[1] == expected_column_count

    expected_columns = ['project_id', 'poly_id', 'site_id', 'project_name', 'geometry', 'plantstart', 'plantend', 'practice', 'target_sys', 'dist', 'project_phase', 'area']
    all_exist = all(col in cleaned_features.columns for col in expected_columns)
    assert all_exist


def get_project_tm_features(project_id):
    # Clear the tm_response outfile path
    PARAMS['outfile']['tm_response'] = ''

    features = tm_pull_wrapper(PARAMS, [project_id])
    return features
