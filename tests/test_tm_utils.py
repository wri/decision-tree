import os

import pandas as pd
from gri_shared_library.os_tools import create_folder

from conftest import DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH
from decision_tree.api_utils import download_geoparquet
from decision_tree.process_api_results import TestProjectHandling
from decision_tree.process_api_results import process_tm_results
from decision_tree.tools import convert_to_os_path, load_secrets, load_yaml
from gri_shared_library.constants_for_tests import TEST_01_GRI_PROJECT_ID
from tools import folder_cleanup

params_path = os.path.join(DT_TEST_PARAMS_DIR, "params_full.yaml")
PARAMS = load_yaml(params_path)
SECRETS = load_secrets(SECRETS_FILE_PATH)


def test_tm_features():
    project_ids = [TEST_01_GRI_PROJECT_ID]
    # pre-run setup
    folder_cleanup(params_path)

    parquet_outfile, features = _get_project_tm_features(project_ids)

    # Confirm that the file contains at least one polygon
    assert len(features) == 3

    # post-run cleanup
    folder_cleanup(params_path)


def test_clean_tm_features():
    project_ids = [TEST_01_GRI_PROJECT_ID]; test_project_handling = TestProjectHandling.ONLY
    # pre-run setup
    folder_cleanup(params_path)

    parquet_outfile, features = _get_project_tm_features(project_ids)

    outfile = PARAMS['outfile']
    project_data_dir = outfile["project_data_folder"]
    geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])

    try:
        cleaned_features = process_tm_results(params=PARAMS, tm_df=features, geojson_dir=geojson_dir,
                                              project_ids=project_ids, test_project_handling=test_project_handling)

        assert len(cleaned_features) == 3

        actual_attribute_count = cleaned_features.shape[1]
        expected_column_count = 14
        assert actual_attribute_count == expected_column_count

        expected_columns = ['cohort', 'project_id', 'poly_id', 'site_id', 'project_name', 'geometry', 'plantstart', 'practice', 'target_sys', 'area', 'notes_base', 'notes_ev']
        all_exist = all(col in cleaned_features.columns for col in expected_columns)
        assert all_exist

    finally:
        # post-run cleanup
        folder_cleanup(params_path)


def _get_project_tm_features(project_ids):
    outfile = PARAMS['outfile']
    data_v = outfile["data_version"]
    project_data_dir = outfile["project_data_folder"]
    parquet_outfile = convert_to_os_path(project_data_dir, outfile['geoparquet'].format(cohort=outfile['cohort'], data_version=data_v))

    # cleanup target
    tm_raw_dir = os.path.dirname(parquet_outfile)
    create_folder(tm_raw_dir)

    download_geoparquet(PARAMS, SECRETS, parquet_outfile)
    # Read parquet with pandas and standardize column names.
    df = pd.read_parquet(parquet_outfile)
    df.columns = df.columns.str.lower()

    # Thin to projects
    features = df[df['project_id'].isin(project_ids)].reset_index(drop=True)

    return parquet_outfile, features

