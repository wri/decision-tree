import pandas as pd

def analyze_image_availability(params, proj_df, maxar_fp):
    """
    Assesses image availability for baseline & early verification per 
    project/polygon based on user defined windows.

    Parameters:
    - proj_df (pd.DataFrame): DataFrame containing project characteristics.
    - img_df (pd.DataFrame): DataFrame containing image observations.
    - param_path points to the params.yaml which contains criteria for the decision

    Returns:
    - pd.DataFrame: Merged DataFrame with image availability counts per polygon.
    """
    n_projects = proj_df['project_id'].nunique()
    n_polys = proj_df['poly_id'].nunique() 
    print(f"Analyzing image availability for {n_projects} projects and {n_polys} polygons...")

    criteria = params.get('criteria', {})
    baseline_range = tuple(criteria.get('baseline_range'))
    ev_range = tuple(criteria.get('ev_range'))
    cloud_thresh = criteria.get('cloud_thresh')
    off_nadir = criteria.get('off_nadir')
    sun = criteria.get('sun_elevation')
    
    proj_df.columns = proj_df.columns.str.lower()
    img_df = pd.read_csv(maxar_fp, dtype={"datetime": "string"})
    img_df = img_df[[
            'project_id', 'poly_id', 'site_id',
            'datetime',
            'area:cloud_cover_percentage',
            'view:sun_elevation',
            'area:avg_off_nadir_angle'
            ]].copy()

    img_df["img_date"] = pd.to_datetime(
        img_df["datetime"].str.strip(), 
        format="%m/%d/%y",
        errors="coerce"
        )

    # Ensure correct datatypes
    num_cols = ['area:cloud_cover_percentage', 'view:sun_elevation', 'area:avg_off_nadir_angle']
    img_df[num_cols] = img_df[num_cols].apply(pd.to_numeric, errors='coerce')
    proj_df['plantstart'] = pd.to_datetime(proj_df['plantstart'], errors='coerce')

    # merge img df and feats
    merged = img_df.merge(proj_df, on=['project_id', 'poly_id'], how='left')

    # Compute image availability window
    merged['date_diff'] = (merged['img_date'] - merged['plantstart']).dt.days

    baseline = merged[
        (merged['date_diff'] >= baseline_range[0]) &
        (merged['date_diff'] <= baseline_range[1]) &
        (merged['area:cloud_cover_percentage'] < cloud_thresh) &
        (merged['view:sun_elevation'] >= sun) &
        (merged['area:avg_off_nadir_angle'] <= off_nadir)
    ]
    baseline_summary = (
        baseline.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='baseline_img_count')
    )
    ev = merged[
        (merged['date_diff'] >= ev_range[0]) &
        (merged['date_diff'] <= ev_range[1]) &
        (merged['area:cloud_cover_percentage'] < cloud_thresh) &
        (merged['view:sun_elevation'] >= sun) &
        (merged['area:avg_off_nadir_angle'] <= off_nadir)
    ]
    ev_summary = (
        ev.groupby(['project_id', 'poly_id'])
        .size()
        .reset_index(name='ev_img_count')
    )
    final_summary = proj_df.merge(baseline_summary, on=['project_id', 'poly_id'], how='left') \
                           .merge(ev_summary, on=['project_id', 'poly_id'], how='left')

    # Fill in missing image counts with 0
    final_summary[['baseline_img_count', 'ev_img_count']] = final_summary[
        ['baseline_img_count', 'ev_img_count']
    ].fillna(0)
    return final_summary