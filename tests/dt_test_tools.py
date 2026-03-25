import math
import os

from tests.conftest import PROJECT_ROOT


def delete_source_geojsons_file(filename):
    scratch_file_path1 = os.path.join(PROJECT_ROOT, "tests", "data", "geojsons", filename)
    if os.path.isfile(scratch_file_path1):
        os.remove(scratch_file_path1)


def has_expected_project_ev_values(slope_statistics, poly_results, prj_results,
                                   expected_project_count, expected_polygon_count, expected_project_label,
                                   expected_median_slope, expected_baseline_total, expected_ev_total):
    assert len(prj_results) == expected_project_count

    assert prj_results['ev_total_poly_count'].values[0] == expected_polygon_count

    actual_project_label = prj_results['baseline_project_label'].values[0]
    assert actual_project_label == expected_project_label, f"Expected: {expected_project_label!r}, Actual: {actual_project_label!r}"

    if expected_median_slope is not None:
        actual_median_slope = slope_statistics['median_slope'].median()
        assert math.isclose(actual_median_slope, expected_median_slope, rel_tol=0.1), f"Expected {expected_median_slope}, got {actual_median_slope}"
    else:
        assert slope_statistics is None, f"Expected slop_statistics to be None, but got values."

    actual_baseline_total = poly_results["baseline_cost"].sum()
    assert actual_baseline_total == expected_baseline_total, f"Expected baseline_total = {expected_baseline_total}, got {actual_baseline_total}"

    actual_ev_total = poly_results["ev_cost"].sum()
    assert actual_ev_total == expected_ev_total, f"Expected ev_total = {expected_ev_total}, got {actual_ev_total}"
