import contextlib
import os
import shutil
from unittest import mock

import pandas as pd
import pytest
from gri_shared_library.os_tools import get_project_root_dir

from decision_tree.tools import load_yaml


def has_expected_project_ev_values(poly_results, prj_results, sample_poly_id,
                                   expected_poly_baseline_suitability, expected_poly_baseline_total, expected_poly_ev_total,
                                   expected_project_count, expected_project_ev_label, expected_pct_area_scored
                                   ):
    # poly_results
    actual_poly_baseline_suitability = str(poly_results[poly_results['poly_id'] == sample_poly_id]['baseline_remote_suitability'].iat[0])
    assert actual_poly_baseline_suitability == str(expected_poly_baseline_suitability),\
        f"Expected baseline_remote_suitability = {expected_poly_baseline_suitability}, got {actual_poly_baseline_suitability}"

    actual_baseline_total = poly_results["baseline_cost"].sum()
    assert actual_baseline_total == pytest.approx(expected_poly_baseline_total, abs=0.1),\
        f"Expected baseline_total = {expected_poly_baseline_total}, got {actual_baseline_total}"

    actual_ev_total = poly_results["ev_cost"].sum()
    assert actual_ev_total == pytest.approx(expected_poly_ev_total, abs=0.1), \
        f"Expected ev_total = {expected_poly_ev_total}, got {actual_ev_total}"


    # prj_results
    actual_project_count = len(prj_results)
    assert actual_project_count == expected_project_count, \
        f"Expected project_count = {expected_project_count}, got {actual_project_count}"

    actual_project_ev_label = prj_results['ev_label'].values[0]
    assert actual_project_ev_label == expected_project_ev_label, \
        f"Expected ev_label = {expected_project_ev_label}, got {actual_project_ev_label}"

    actual_pct_area_scored = prj_results['ev_pct_area_scored'].values[0]
    assert actual_pct_area_scored == pytest.approx(expected_pct_area_scored, abs=0.1),\
        f"Expected pct_area_scored: {expected_pct_area_scored!r}, Actual: {actual_pct_area_scored!r}"


def _get_folders(directory, exclude):
    exclude = set(exclude)
    return [
        entry.name
        for entry in os.scandir(directory)
        if entry.is_dir() and entry.name not in exclude
    ]


TEST_DATA_ROOT_PARTS = ("tests", "data")


def _resolve_cleanup_dir(params_path):
    """
    Resolve the project_data_folder from a params file and refuse anything that is not
    strictly inside tests/data, so a mistyped params file cannot wipe real data.
    """
    params = load_yaml(params_path)
    root = get_project_root_dir()
    allowed = os.path.realpath(os.path.join(root, *TEST_DATA_ROOT_PARTS))
    data_dir = os.path.realpath(os.path.join(root, params['outfile']['project_data_folder']))
    try:
        inside = os.path.commonpath([data_dir, allowed]) == allowed and data_dir != allowed
    except ValueError:  # e.g. different drives on Windows
        inside = False
    if not inside:
        raise ValueError(f"Refusing to clean '{data_dir}': project_data_folder in {params_path} "
                         f"must be a sub-folder of '{allowed}'.")
    return data_dir


def folder_cleanup(params_path):
    data_dir = _resolve_cleanup_dir(params_path)
    keeper_folders = ["imagery_availability", "tree_output"]
    scratch_folders = _get_folders(data_dir, keeper_folders)
    for folder in scratch_folders:
        folder_dir = os.path.join(data_dir, folder)
        shutil.rmtree(folder_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# RDS helpers for integration tests
# ---------------------------------------------------------------------------
def fetch_rds_rows(workflow, table_key: str, project_ids: list[str], run_datetime) -> pd.DataFrame:
    """Rows of the RDS table params rds.<table_key> for these projects at one run_datetime."""
    from psycopg import sql
    from decision_tree.rds_dao import rds_connection

    settings, connection = workflow._rds_config()
    query = sql.SQL("SELECT * FROM {}.{} WHERE project_id = ANY(%s) AND run_datetime = %s").format(
        sql.Identifier(settings["schema"]), sql.Identifier(settings[table_key]))
    with rds_connection(settings["database"], env=settings["env"], **connection) as conn:
        cur = conn.execute(query, (list(project_ids), run_datetime))
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


@contextlib.contextmanager
def rds_rows_cleanup(workflow, project_ids: list[str]):
    """Record every run_datetime the pipeline uses inside the block; on exit, delete the rows
    those runs wrote for these projects from the RDS polygon and project tables.

    Yields the list of recorded run_datetimes (in call order). Only rows matching both the
    project ids and a recorded run_datetime are deleted, so other runs' data is untouched.
    """
    import decision_tree.run_decision_tree as rdt
    from psycopg import sql
    from decision_tree.rds_dao import rds_connection

    run_datetimes: list = []
    real_now = rdt._now_utc

    def recording_now():
        value = real_now()
        run_datetimes.append(value)
        return value

    try:
        with mock.patch.object(rdt, "_now_utc", side_effect=recording_now):
            yield run_datetimes
    finally:
        if run_datetimes:
            settings, connection = workflow._rds_config()
            with rds_connection(settings["database"], env=settings["env"], **connection) as conn:
                with conn.transaction():
                    for table_key in ("poly_table", "prj_table"):
                        conn.execute(
                            sql.SQL("DELETE FROM {}.{} WHERE project_id = ANY(%s) AND run_datetime = ANY(%s)")
                            .format(sql.Identifier(settings["schema"]), sql.Identifier(settings[table_key])),
                            (list(project_ids), list(set(run_datetimes))),
                        )


def delete_rds_project_rows(workflow, project_ids: list[str]) -> None:
    """Delete every row for these projects from the RDS polygon and project tables.

    Only for test-only project ids that no real run uses.
    """
    from psycopg import sql
    from decision_tree.rds_dao import rds_connection

    settings, connection = workflow._rds_config()
    with rds_connection(settings["database"], env=settings["env"], **connection) as conn:
        with conn.transaction():
            for table_key in ("poly_table", "prj_table"):
                conn.execute(
                    sql.SQL("DELETE FROM {}.{} WHERE project_id = ANY(%s)")
                    .format(sql.Identifier(settings["schema"]), sql.Identifier(settings[table_key])),
                    (list(project_ids),),
                )

