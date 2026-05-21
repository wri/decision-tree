
def has_expected_project_ev_values(poly_results, prj_results,
                                   expected_poly_baseline_suitability, expected_poly_baseline_total, expected_poly_ev_total,
                                   expected_project_count, expected_project_ev_label, expected_pct_area_scored
                                   ):
    # poly_results
    actual_poly_baseline_suitability = str(poly_results['baseline_remote_suitability'].values[0])
    assert actual_poly_baseline_suitability == str(expected_poly_baseline_suitability),\
        f"Expected baseline_remote_suitability = {actual_poly_baseline_suitability}, got {expected_poly_baseline_suitability}"

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
