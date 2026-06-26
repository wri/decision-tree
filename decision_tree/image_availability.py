
import pandas as pd

from decision_tree.tools import resolve_indicator_window_range


def analyze_image_availability(params,
                               proj_df, 
                               maxar_fp: str):
    """
    Assesses image availability for baseline & early verification per 
    project/polygon based on user defined windows.

    Parameters:
    - params - string path to the params.yaml which contains criteria for the decision
    - proj_df (pd.DataFrame): DataFrame containing project characteristics.
    - maxar_fp - string path to the maxar_fp

    Returns:
    - pd.DataFrame: Merged DataFrame with image availability counts per polygon.
    """
    n_projects = proj_df['project_id'].nunique()
    n_polys = proj_df['poly_id'].nunique() 
    print(f"Analyzing image availability for {n_projects} projects and {n_polys} polygons...")

    baseline_range = resolve_indicator_window_range(params, 'BASELINE')
    ext_baseline_range = resolve_indicator_window_range(params, 'EXT_BASELINE')
    ev_range = resolve_indicator_window_range(params, 'EARLY_INSIGHTS')
    
    proj_df.columns = proj_df.columns.str.lower()
    # Create dataFrame containing image observations.
    img_df = pd.read_csv(maxar_fp, dtype={"datetime": "string"})
    img_df = img_df[[
            'project_id', 'poly_id', 'site_id',
            'datetime',
            'area:cloud_cover_percentage',
            'view:sun_elevation',
            'area:avg_off_nadir_angle',
            ]].copy()

    img_df["img_date"] = pd.to_datetime(
        img_df["datetime"].str.strip(),
        format="%Y-%m-%d %H:%M:%S.%f%z",
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)

    # Ensure correct datatypes & merge
    proj_df['plantstart'] = pd.to_datetime(proj_df['plantstart'], errors='coerce')
    merged = img_df.merge(proj_df, on=['project_id', 'poly_id'], how='left')
    # add step here to check if any rows were dropped

    # Compute image availability window
    merged['date_diff'] = (merged['img_date'] - merged['plantstart']).dt.days

    baseline = merged[
        (merged['date_diff'] >= baseline_range[0]) &
        (merged['date_diff'] <= baseline_range[1])
    ]
    baseline_summary = (
        baseline.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='baseline_img_count')
    )
    baseline_ext = merged[
        (merged['date_diff'] >= ext_baseline_range[0]) &
        (merged['date_diff'] <= ext_baseline_range[1])
    ]
    baseline_ext_summary = (
        baseline_ext.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='baseline_ext_img_count')
    )

    ev = merged[
        (merged['date_diff'] >= ev_range[0]) &
        (merged['date_diff'] <= ev_range[1])
    ]
    ev_summary = (
        ev.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='ev_img_count')
    )
    final_summary = proj_df \
    .merge(baseline_summary,     on=['project_id', 'poly_id'], how='left') \
    .merge(baseline_ext_summary, on=['project_id', 'poly_id'], how='left') \
    .merge(ev_summary,           on=['project_id', 'poly_id'], how='left')

    final_summary[['baseline_img_count', 'baseline_ext_img_count', 'ev_img_count']] = \
        final_summary[['baseline_img_count', 'baseline_ext_img_count', 'ev_img_count']].fillna(0)
    
    return final_summary