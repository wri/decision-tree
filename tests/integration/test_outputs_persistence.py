"""Integration test (needs AWS access): where the pipeline's outputs are persisted.

- Intermediate files (feats, slope_stats, tree_results) are written locally when
  params outfile.save_local_intermediates is true.
- Final polygon and project results are written to the RDS tables when params rds.upload is true.

Teardown removes the local files and deletes exactly the RDS rows these runs wrote.
Needs secrets.yaml aws.bastion_id_for_rds and aws.land_aws_profile.
"""
import os

import pandas as pd
import pytest

from decision_tree.constants import TestProjectHandling
from decision_tree.run_decision_tree import main
from gri_shared_library.constants_for_tests import TEST_01_GRI_PROJECT_ID

from tests.conftest import DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH
from tests.tools import fetch_rds_rows, folder_cleanup, rds_rows_cleanup

PARAMS_PATH = os.path.join(DT_TEST_PARAMS_DIR, "params_projectids_outputs.yaml")
PROJECT_IDS = [TEST_01_GRI_PROJECT_ID]


@pytest.fixture
def workflow():
    """Workflow for the outputs params file; removes every local file it writes on teardown."""
    folder_cleanup(PARAMS_PATH)
    wf = main(PARAMS_PATH, SECRETS_FILE_PATH, parse_only=True)
    yield wf
    # tree_output/ is kept by folder_cleanup (it holds committed fixtures), so remove ours explicitly
    if os.path.exists(wf.tree_results):
        os.remove(wf.tree_results)
    folder_cleanup(PARAMS_PATH)


def _assert_non_empty_csv(path: str) -> pd.DataFrame:
    assert os.path.isfile(path), f"expected file not written: {path}"
    df = pd.read_csv(path)
    assert len(df) > 0, f"file is empty: {path}"
    return df


@pytest.mark.network
def test_intermediates_saved_locally_and_results_saved_to_rds(workflow):
    assert workflow.params["outfile"]["save_local_intermediates"] is True
    assert workflow.params["rds"]["upload"] is True
    intermediates = {"feats": workflow.feats, "slope_stats": workflow.slope_stats,
                     "tree_results": workflow.tree_results}
    for key, path in intermediates.items():
        assert not os.path.exists(path), f"{key} left over from an earlier run: {path}"

    with rds_rows_cleanup(workflow, PROJECT_IDS) as run_datetimes:
        poly_results = workflow.run_polygon_metrics(project_ids=PROJECT_IDS,
                                                    test_project_handling=TestProjectHandling.ONLY)
        prj_results = workflow.run_project_metrics(project_ids=PROJECT_IDS)  # reads polygons from RDS
        poly_run, prj_run = run_datetimes[0], run_datetimes[-1]

        # Intermediate files written locally
        feats = _assert_non_empty_csv(intermediates["feats"])
        _assert_non_empty_csv(intermediates["slope_stats"])
        tree_results = _assert_non_empty_csv(intermediates["tree_results"])
        assert set(feats["project_id"].astype(str)) == set(PROJECT_IDS)
        assert set(tree_results["poly_id"].astype(str)) == set(poly_results["poly_id"].astype(str))

        # Final results written to RDS: one polygon row per polygon, one project row
        poly_rows = fetch_rds_rows(workflow, "poly_table", PROJECT_IDS, poly_run)
        assert len(poly_rows) == len(poly_results)
        assert set(poly_rows["poly_id"]) == set(poly_results["poly_id"].astype(str))
        assert set(poly_rows["cohort"].dropna()) <= {workflow.cohort, "terrafund-cohort-1", "terrafund-cohort-2"}

        prj_rows = fetch_rds_rows(workflow, "prj_table", PROJECT_IDS, prj_run)
        assert len(prj_rows) == 1
        assert prj_rows["project_id"].iat[0] == TEST_01_GRI_PROJECT_ID
        assert prj_rows["base_label"].iat[0] == prj_results["base_label"].iat[0]
        assert prj_rows["ev_label"].iat[0] == prj_results["ev_label"].iat[0]

    # Teardown of the RDS rows really removed them
    assert fetch_rds_rows(workflow, "poly_table", PROJECT_IDS, poly_run).empty
    assert fetch_rds_rows(workflow, "prj_table", PROJECT_IDS, prj_run).empty
