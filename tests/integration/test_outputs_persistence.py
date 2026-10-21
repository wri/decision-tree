"""Integration test (needs AWS access): where the pipeline's outputs are persisted.

- Intermediate files (feats, slope_stats, tree_results) are written locally when
  params outfile.save_local_intermediates is true.
- Final polygon and project results are written to the RDS tables when params rds.upload is true.

Teardown removes the local files and deletes exactly the RDS rows these runs wrote.
Needs secrets.yaml aws.bastion_id_for_rds and aws.land_aws_profile.
"""
import datetime
import os
import uuid
from unittest import mock

import pandas as pd
import pytest

import decision_tree.rds_dao as rds_dao
from decision_tree.constants import TestProjectHandling
from decision_tree.run_decision_tree import compute_project_results, main
from gri_shared_library.constants_for_tests import TEST_01_GRI_PROJECT_ID

from tests.conftest import DT_TEST_PARAMS_DIR, SECRETS_FILE_PATH
from tests.tools import delete_rds_project_rows, fetch_rds_rows, folder_cleanup, rds_rows_cleanup

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


# ---------------------------------------------------------------------------
# Project step uses the latest row of every polygon, not just the latest run's
# ---------------------------------------------------------------------------
RDS_PARAMS_PATH = os.path.join(DT_TEST_PARAMS_DIR, "params_projectids_rds.yaml")


@pytest.fixture
def synthetic_project():
    """A made-up project id unique to this test run; all its RDS rows are deleted on teardown."""
    folder_cleanup(RDS_PARAMS_PATH)
    wf = main(RDS_PARAMS_PATH, SECRETS_FILE_PATH, parse_only=True)
    project_id = f"test-latest-polygons-{uuid.uuid4().hex[:12]}"
    yield wf, project_id
    delete_rds_project_rows(wf, [project_id])
    folder_cleanup(RDS_PARAMS_PATH)


def _synthetic_polygons(project_id: str) -> pd.DataFrame:
    """Five polygons with the columns the project aggregation needs."""
    return pd.DataFrame({
        "project_id": project_id,
        "poly_id": [f"{project_id}-poly-{i}" for i in range(5)],
        "project_name": "TEST_LATEST_POLYGONS",
        "target_sys": "agroforest",
        "area": [5.0, 3.0, 1.0, 0.5, 2.0],
        "baseline_decision": ["weak remote", "strong remote", "weak field", "weak remote", "strong remote"],
        "ev_decision": ["weak remote", "strong remote", "weak field", "weak remote", "strong remote"],
        "baseline_remote_suitability": 60.0,
        "ev_remote_suitability": 60.0,
    })


@pytest.mark.network
def test_project_step_uses_latest_row_of_every_polygon(synthetic_project):
    workflow, project_id = synthetic_project
    settings, connection = workflow._rds_config()
    table_settings = {k: settings[k] for k in ("env", "database", "schema", "poly_table", "prj_table")}

    def upload(polygons, run_datetime):
        rds_dao.upload_results(poly_results=polygons, run_datetime=run_datetime, cohort=workflow.cohort,
                               **table_settings, **connection)

    run1 = datetime.datetime(2000, 1, 1, 9)
    run2 = datetime.datetime(2000, 1, 1, 10)
    polygons = _synthetic_polygons(project_id)

    # Run 1: all five polygons
    upload(polygons, run1)
    # Run 2: only two of them, with changed values
    rerun = polygons.iloc[:2].copy()
    rerun["baseline_decision"] = rerun["ev_decision"] = "strong field"
    rerun["baseline_remote_suitability"] = rerun["ev_remote_suitability"] = 10.0
    upload(rerun, run2)

    # Record the polygon rows the project step reads from RDS
    read_back = {}
    real_read = rds_dao.read_latest_poly_results

    def recording_read(**kwargs):
        read_back["polygons"] = real_read(**kwargs)
        return read_back["polygons"]

    with mock.patch.object(rds_dao, "read_latest_poly_results", side_effect=recording_read):
        prj_results = workflow.run_project_metrics(project_ids=[project_id])
    used = read_back["polygons"]

    # Every polygon of the project is used, each at its newest version
    assert sorted(used["poly_id"]) == sorted(polygons["poly_id"])
    used = used.set_index("poly_id")
    assert (used.loc[rerun["poly_id"], "run_datetime"] == run2).all()
    untouched = polygons.loc[~polygons["poly_id"].isin(rerun["poly_id"]), "poly_id"]
    assert (used.loc[untouched, "run_datetime"] == run1).all()
    assert (used.loc[rerun["poly_id"], "baseline_decision"] == "strong field").all()

    # The project result is the aggregation of exactly those rows
    expected_polygons = pd.concat([rerun, polygons[polygons["poly_id"].isin(untouched)]])
    expected = compute_project_results(workflow.params, expected_polygons)
    pd.testing.assert_frame_equal(prj_results.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, rtol=1e-5)

