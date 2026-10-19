"""
Offline tests for the polygon / project step split in run_decision_tree.py.

No network or credentials are needed: all output paths are redirected to tmp_path
and every data-acquisition call is mocked.

Covers:
  - The project step never acquires inputs, whatever the mode
  - The polygon step writes polygon output only
  - Mode / project_id validation in the polygon step
  - Fail-fast when Maxar metadata is missing
  - main() step dispatch
  - RDS upload flag
  - Saved polygon output round-trips to the same project result as in-memory data
"""

import datetime
import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from conftest import DT_TEST_PARAMS_DIR
import decision_tree.run_decision_tree as rdt
from decision_tree.polygon_decisions import TCC_BASELINE_OFFSET_YEARS, TCC_EI_OFFSET_YEARS
from decision_tree.run_decision_tree import VerificationDecisionTree, compute_project_results

PARAMS_SCORE = os.path.join(DT_TEST_PARAMS_DIR, "params_score.yaml")
ACQUISITION_CALLS = ["download_geoparquet", "copernicus_pull_wrapper", "get_tm_feats",
                     "get_project_ids_from_geoparquet", "compute_branches"]


@pytest.fixture
def workflow(tmp_path):
    """A score-mode workflow whose input and output files all live in tmp_path."""
    wf = VerificationDecisionTree(PARAMS_SCORE, secrets_path=None)
    for key in wf.checkpoint.paths:
        path = str(tmp_path / f"{key}.csv")
        wf.checkpoint.paths[key] = path
        setattr(wf, key, path)
    wf.maxar_meta = str(tmp_path / "maxar_meta.csv")
    return wf


@pytest.fixture
def no_acquisition():
    """Patch every data-acquisition call so a test can assert none was made."""
    with mock.patch.multiple(rdt, **{name: mock.DEFAULT for name in ACQUISITION_CALLS}) as mocks:
        with mock.patch.object(rdt.clean, "_read_geoparquet") as read_gp:
            mocks["_read_geoparquet"] = read_gp
            yield mocks


def _tree_results() -> pd.DataFrame:
    """Realistic decision-tree output: two projects, mixed decisions, a mangrove, gaps."""
    base_year = 2022
    ttc_base = f"ttc_{base_year + TCC_BASELINE_OFFSET_YEARS}"
    ttc_ev = f"ttc_{base_year + TCC_EI_OFFSET_YEARS}"
    rows = [
        # project_id, decision, target_sys, area, ttc_base, ttc_ev, img_1yr, img_ext, img_ev, slope_area
        ("p1", "weak remote",     "agroforest", 5.0, 20.0, 30.0, 3, 4, 2, 10.0),
        ("p1", "strong remote",   "agroforest", 3.0, 10.0, 15.0, 4, 4, 4, np.nan),
        ("p1", "weak field",      "agroforest", 1.0, 80.0, 85.0, 0, 1, 0, 60.0),
        ("p1", "mangrove",        "mangrove",   0.5, 50.0, 50.0, 1, 1, 1, 0.0),
        ("p1", None,              "agroforest", 0.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("p2", "strong field",    "riparian-area-or-wetland", 2.0, 90.0, 95.0, 0, 0, 0, 70.0),
        ("p2", "review required", "riparian-area-or-wetland", 1.0, np.nan, np.nan, 0, 0, 0, 40.0),
        ("p2", "strong field",    "riparian-area-or-wetland", 4.0, 85.0, np.nan, 1, 1, np.nan, 55.0),
    ]
    df = pd.DataFrame(rows, columns=["project_id", "baseline_decision", "target_sys", "area",
                                     ttc_base, ttc_ev, "baseline_img_count", "baseline_ext_img_count",
                                     "ev_img_count", "slope_area"])
    df["ev_decision"] = df["baseline_decision"]
    df["poly_id"] = [f"poly_{i}" for i in range(len(df))]
    df["project_name"] = "name_" + df["project_id"]
    df["baseline_year"] = base_year
    df["notes_base"] = ""
    df["notes_ev"] = ""
    return df


PROJECT_IDS = ["p1", "p2"]
RDS_SECRETS = {"aws": {"bastion_id_for_rds": "i-test", "land_aws_profile": "p"}}


def _poly() -> pd.DataFrame:
    """Polygon results with the columns the project step needs."""
    return _tree_results().assign(baseline_remote_suitability=50.0, ev_remote_suitability=50.0)


def _poly_from_rds() -> pd.DataFrame:
    """Polygon results as read back from RDS (with run_datetime)."""
    return _poly().assign(run_datetime=pd.Timestamp("2026-01-01"))


# ---------------------------------------------------------------------------
# Project step
# ---------------------------------------------------------------------------
class TestProjectStep:
    @pytest.mark.parametrize("mode", ["full", "projectids", "score"])
    def test_does_not_acquire_inputs(self, workflow, no_acquisition, mode):
        workflow.mode = mode
        workflow.secrets = RDS_SECRETS
        with mock.patch("decision_tree.rds_dao.read_latest_poly_results",
                        return_value=_poly_from_rds()):
            prj = workflow.run_project_metrics(project_ids=PROJECT_IDS)

        assert len(prj) == 2
        for name, m in no_acquisition.items():
            assert not m.called, f"project step called {name}"

    @pytest.mark.parametrize("project_ids", [None, [], "p1"])
    def test_project_ids_are_required(self, workflow, project_ids):
        with pytest.raises(ValueError, match="project_ids is required"):
            workflow.run_project_metrics(project_ids=project_ids, poly_results=_poly())

    def test_writes_project_output(self, workflow):
        workflow.run_project_metrics(PROJECT_IDS, poly_results=_poly())
        assert os.path.exists(workflow.prj_score)

    def test_supplied_poly_results_skip_rds(self, workflow):
        with mock.patch("decision_tree.rds_dao.read_latest_poly_results") as read:
            prj = workflow.run_project_metrics(["p2"], poly_results=_poly())
        assert not read.called
        assert prj["project_id"].tolist() == ["p2"]

    def test_unknown_project_in_supplied_poly_results_raises(self, workflow):
        with pytest.raises(ValueError, match="not found in polygon results"):
            workflow.run_project_metrics(["p1", "p9"], poly_results=_poly())

    def test_reads_latest_poly_results_from_rds_for_the_project_ids(self, workflow):
        # reading does not depend on rds.upload, which only controls saving
        assert workflow.params["rds"]["upload"] is False
        workflow.secrets = RDS_SECRETS
        with mock.patch("decision_tree.rds_dao.read_latest_poly_results",
                        return_value=_poly_from_rds()) as read, \
             mock.patch("decision_tree.rds_dao.upload_results") as upload, \
             mock.patch.object(workflow.checkpoint, "load") as csv_load:
            prj = workflow.run_project_metrics(project_ids=PROJECT_IDS)
        assert not csv_load.called
        assert not upload.called  # rds.upload is false
        kwargs = read.call_args.kwargs
        assert kwargs["project_ids"] == PROJECT_IDS
        assert kwargs["poly_table"] == workflow.params["rds"]["poly_table"]
        assert kwargs["bastion_id"] == "i-test" and kwargs["aws_profile"] == "p"
        assert sorted(prj["project_id"]) == PROJECT_IDS

    def test_project_missing_from_rds_raises(self, workflow):
        workflow.secrets = RDS_SECRETS
        only_p1 = _poly_from_rds().query("project_id == 'p1'")
        with mock.patch("decision_tree.rds_dao.read_latest_poly_results", return_value=only_p1):
            with pytest.raises(ValueError, match=r"No polygon results in .* for project id\(s\): \['p2'\]"):
                workflow.run_project_metrics(project_ids=PROJECT_IDS)

    def test_rds_read_requires_bastion_id(self, workflow):
        workflow.secrets = {"aws": {"land_aws_profile": "p"}}
        with pytest.raises(ValueError, match="bastion_id_for_rds"):
            workflow.run_project_metrics(project_ids=PROJECT_IDS)

    def test_rds_upload_uses_params_and_secrets(self, workflow):
        workflow.params["rds"]["upload"] = True
        workflow.secrets = RDS_SECRETS
        summary = SimpleNamespace(poly_rows=0, prj_rows=2, run_datetime=datetime.datetime(2026, 1, 1))
        with mock.patch("decision_tree.rds_dao.upload_results", return_value=summary) as upload:
            prj = workflow.run_project_metrics(PROJECT_IDS, poly_results=_poly())
        kwargs = upload.call_args.kwargs
        assert kwargs["prj_results"] is prj and kwargs["poly_results"] is None
        assert kwargs["bastion_id"] == "i-test" and kwargs["aws_profile"] == "p"
        assert "region" not in kwargs  # region comes from land_aws_profile
        for key in ("env", "database", "schema", "poly_table", "prj_table"):
            assert kwargs[key] == workflow.params["rds"][key]

    def test_rds_upload_requires_bastion_id(self, workflow):
        workflow.params["rds"]["upload"] = True
        workflow.secrets = {"aws": {"land_aws_profile": "p"}}
        with pytest.raises(ValueError, match="bastion_id_for_rds"):
            workflow.run_project_metrics(PROJECT_IDS, poly_results=_poly())


# ---------------------------------------------------------------------------
# Polygon step
# ---------------------------------------------------------------------------
class TestPolygonStep:
    def test_writes_polygon_output_only(self, workflow):
        _tree_results().to_csv(workflow.tree_results, index=False)
        poly = workflow.run_polygon_metrics()
        assert os.path.exists(workflow.poly_score)
        assert not os.path.exists(workflow.prj_score)
        for col in ["baseline_remote_suitability", "ev_remote_suitability", "baseline_cost", "ev_cost"]:
            assert col in poly.columns

    @pytest.mark.parametrize("save", [False, True])
    def test_intermediates_follow_save_local_intermediates(self, tmp_path, save):
        params = rdt.load_yaml(PARAMS_SCORE)
        params["outfile"]["save_local_intermediates"] = save
        with mock.patch.object(rdt, "load_yaml", return_value=params):
            wf = VerificationDecisionTree(PARAMS_SCORE, secrets_path=None)
        assert wf.checkpoint.enabled is save

    @pytest.mark.parametrize("value", [None, "yes"])
    def test_save_local_intermediates_must_be_boolean(self, value):
        params = rdt.load_yaml(PARAMS_SCORE)
        params["outfile"]["save_local_intermediates"] = value
        with mock.patch.object(rdt, "load_yaml", return_value=params):
            with pytest.raises(ValueError, match="save_local_intermediates"):
                VerificationDecisionTree(PARAMS_SCORE, secrets_path=None)

    def test_notes_columns_follow_their_decisions(self, workflow):
        _tree_results().to_csv(workflow.tree_results, index=False)
        cols = list(workflow.run_polygon_metrics().columns)
        assert cols.index("notes_base") == cols.index("baseline_decision") + 1
        assert cols.index("notes_ev") == cols.index("ev_decision") + 1

    @pytest.mark.parametrize("mode, project_ids, message", [
        ("full", ["p1"], "cannot be specified for the 'full' mode"),
        ("score", ["p1"], "cannot be specified for the 'score' mode"),
        ("projectids", None, "must be specified for 'projectids' mode"),
        ("projectids", [], "must be specified for 'projectids' mode"),
        ("partial", None, "Invalid mode"),
    ])
    def test_mode_validation(self, workflow, no_acquisition, mode, project_ids, message):
        workflow.mode = mode
        with pytest.raises(ValueError, match=message):
            workflow.run_polygon_metrics(project_ids)
        no_acquisition["download_geoparquet"].assert_not_called()

    @pytest.mark.parametrize("mode, project_ids", [("full", None), ("projectids", ["p1"])])
    def test_missing_maxar_metadata_fails_before_download(self, workflow, no_acquisition, mode, project_ids):
        workflow.mode = mode
        assert not os.path.exists(workflow.maxar_meta)
        with pytest.raises(FileNotFoundError, match="Maxar image metadata not found"):
            workflow.run_polygon_metrics(project_ids)
        no_acquisition["download_geoparquet"].assert_not_called()

    def test_score_mode_does_not_require_maxar_metadata(self, workflow, no_acquisition):
        _tree_results().to_csv(workflow.tree_results, index=False)
        workflow.run_polygon_metrics()
        no_acquisition["download_geoparquet"].assert_not_called()


# ---------------------------------------------------------------------------
# Hand-off between steps
# ---------------------------------------------------------------------------
class TestHandOff:
    def test_polygon_output_read_back_from_rds_gives_same_project_result(self, workflow):
        _tree_results().to_csv(workflow.tree_results, index=False)
        workflow.secrets = RDS_SECRETS
        poly_in_memory = workflow.run_polygon_metrics()

        with mock.patch("decision_tree.rds_dao.read_latest_poly_results",
                        return_value=poly_in_memory.assign(run_datetime=pd.Timestamp("2026-01-01"))):
            prj_from_rds = workflow.run_project_metrics(project_ids=PROJECT_IDS)
        prj_in_memory = compute_project_results(workflow.params, poly_in_memory)

        assert set(prj_from_rds["base_label"]) != {"not available"}, "fixture should produce real labels"
        pd.testing.assert_frame_equal(prj_from_rds.reset_index(drop=True),
                                      prj_in_memory.reset_index(drop=True),
                                      check_dtype=False)

    def test_missing_flags_survive_csv_round_trip(self, workflow):
        _tree_results().to_csv(workflow.tree_results, index=False)
        workflow.run_polygon_metrics()
        reloaded = workflow.checkpoint.load("poly_score")
        for col in ["baseline_canopy_missing", "ev_canopy_missing", "baseline_img_missing",
                    "ev_img_missing", "slope_missing"]:
            assert reloaded[col].dtype == bool, f"{col} reloaded as {reloaded[col].dtype}"

    def test_project_subset(self, workflow):
        _tree_results().to_csv(workflow.tree_results, index=False)
        poly = workflow.run_polygon_metrics()
        prj = workflow.run_project_metrics(project_ids=["p2"], poly_results=poly)
        assert prj["project_id"].tolist() == ["p2"]


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------
class TestMainDispatch:
    @pytest.mark.parametrize("step, expected", [
        ("polygon", {"run_polygon_metrics"}),
        ("project", {"run_project_metrics"}),
        ("all", {"run_polygon_metrics", "run_project_metrics"}),
    ])
    def test_step_calls_only_its_methods(self, step, expected):
        methods = ["run_polygon_metrics", "run_project_metrics"]
        with mock.patch.multiple(VerificationDecisionTree, **{m: mock.DEFAULT for m in methods}) as mocks:
            rdt.main(PARAMS_SCORE, None, step=step, project_ids=["p1"])
        for name, m in mocks.items():
            assert m.called == (name in expected), f"step={step}: {name} called={m.called}"

    def test_all_passes_in_memory_polygon_results_to_project_step(self):
        sentinel = pd.DataFrame({"project_id": ["p2", "p1", "p2"]})
        with mock.patch.object(VerificationDecisionTree, "run_polygon_metrics", return_value=sentinel) as poly, \
             mock.patch.object(VerificationDecisionTree, "run_project_metrics") as prj:
            rdt.main(PARAMS_SCORE, None, step="all", project_ids=None)
        poly.assert_called_once_with(None)
        # no project ids given (full/score mode): every project the polygon step produced
        prj.assert_called_once_with(project_ids=["p1", "p2"], poly_results=sentinel)

    def test_project_step_requires_project_ids(self):
        with pytest.raises(ValueError, match="project_ids is required"):
            rdt.main(PARAMS_SCORE, None, step="project", project_ids=None)

    def test_project_ids_are_passed_through(self):
        with mock.patch.object(VerificationDecisionTree, "run_project_metrics") as m:
            rdt.main(PARAMS_SCORE, None, step="project", project_ids=["p1"])
        m.assert_called_once_with(project_ids=["p1"])

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError, match="Invalid step"):
            rdt.main(PARAMS_SCORE, None, step="bogus")

    def test_parse_only_returns_workflow(self):
        assert isinstance(rdt.main(PARAMS_SCORE, None, parse_only=True), VerificationDecisionTree)
