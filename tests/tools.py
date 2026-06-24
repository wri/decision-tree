import os
import shutil

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
    assert actual_baseline_total == expected_poly_baseline_total,\
        f"Expected baseline_total = {expected_poly_baseline_total}, got {actual_baseline_total}"

    actual_ev_total = poly_results["ev_cost"].sum()
    assert actual_ev_total == expected_poly_ev_total, f"Expected ev_total = {expected_poly_ev_total}, got {actual_ev_total}"


    # prj_results
    actual_project_count = len(prj_results)
    assert actual_project_count == expected_project_count, \
        f"Expected project_count = {expected_project_count}, got {actual_project_count}"

    actual_project_ev_label = prj_results['ev_label'].values[0]
    assert actual_project_ev_label == expected_project_ev_label, \
        f"Expected ev_label = {actual_project_ev_label}, got {expected_project_ev_label}"

    actual_pct_area_scored = prj_results['ev_pct_area_scored'].values[0]
    assert actual_pct_area_scored == expected_pct_area_scored,\
        f"Expected: {expected_pct_area_scored!r}, Actual: {actual_pct_area_scored!r}"


def _get_folders(directory, exclude):
    exclude = set(exclude)
    return [
        entry.name
        for entry in os.scandir(directory)
        if entry.is_dir() and entry.name not in exclude
    ]


def folder_cleanup(params_path):
    params = load_yaml(params_path)
    data_dir = os.path.join(get_project_root_dir(), params['outfile']['project_data_folder'])
    keeper_folders = ["feats", "geojsons", "imagery_availability", "tree_output"]
    scratch_folders = _get_folders(data_dir, keeper_folders)
    for folder in scratch_folders:
        folder_dir = os.path.join(data_dir, folder)
        shutil.rmtree(folder_dir, ignore_errors=True)
