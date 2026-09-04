import json
import os
import tempfile

import pandas as pd

from decision_tree.tools import resolve_indicator_window_range, append_note
from gri_shared_library.s3_tools import download_s3_file


def _explode_project_parquet(parquet_path, project_id):
    """
    Turn one project's baseline search_results parquet into one row per
    (polygon, image) match — the same granularity the old maxar CSV had.

    Parameters:
    - parquet_path (str): local path to a downloaded search_results_baseline.parquet.
    - project_id (str): project_id to stamp onto every resulting row (the parquet
      itself has no project_id column — it's implicit in the S3 folder it came from).

    Returns:
    - pd.DataFrame with columns matching the old maxar CSV's per-image shape:
      project_id, poly_id, datetime, area:cloud_cover_percentage, view:sun_elevation,
      area:avg_off_nadir_angle. (site_id is intentionally omitted — traced through the
      old code and confirmed it never survives into final_summary, so it's dead.)
    """
    img_catalog = pd.read_parquet(parquet_path)[['datetime', 'view:sun_elevation', 'aoi_stats']]

    # aoi_stats is an array of {poly_uuid, cloud_cover, off_nadir, coverage} per image —
    # one entry per polygon that image matches. Explode + json_normalize turns this into
    # one row per (image, polygon) pair, vectorized rather than a manual python loop.
    exploded = img_catalog.explode('aoi_stats').reset_index(drop=True)
    aoi = pd.json_normalize(exploded['aoi_stats'])
    exploded = pd.concat([exploded.drop(columns=['aoi_stats']), aoi], axis=1)
    exploded['project_id'] = project_id

    return exploded.rename(columns={
        'poly_uuid': 'poly_id',
        'cloud_cover': 'area:cloud_cover_percentage',  # per-polygon value, not the image-wide eo:cloud_cover
        'off_nadir': 'area:avg_off_nadir_angle',
    })[[
        'project_id', 'poly_id', 'datetime',
        'area:cloud_cover_percentage', 'view:sun_elevation', 'area:avg_off_nadir_angle',
    ]]


def build_maxar_meta_from_s3(params, proj_df, maxar_fp):
    """
    Download each project's baseline search results from S3, explode them into
    per-(polygon, image) rows, combine across all projects, and save the result
    to `maxar_fp` — the same combined-CSV role the manually-copied maxar file
    used to play.

    Parameters:
    - params - dict loaded from params.yaml.
    - proj_df (pd.DataFrame): must contain project_id and project_name (the
      project's short_name, which is also its S3 folder name).
    - maxar_fp (str): path the combined CSV snapshot is saved to.

    Returns:
    - list: project_ids with no baseline S3 search results yet (not a data error —
      likely just not searched yet). Used by analyze_image_availability to flag
      'no-s3-imagery-data' on those polygons rather than silently reading as 0.

    NOTE — prototype/draft: params.yaml doesn't yet have the S3 bucket/prefix/profile
    keys this reads (params['s3']['vhrimg_bucket'], ['vhrimg_prefix'], ['profile_name'])

    """
    s3_cfg = params['s3']
    bucket = s3_cfg['vhr_img']      
    profile_name = s3_cfg.get('profile_name')  # TODO: confirm which AWS profile

    projects = proj_df[['project_id', 'project_name']].drop_duplicates()

    combined = []
    failed_project_ids = []

    with tempfile.TemporaryDirectory() as scratch_dir:
        for _, row in projects.iterrows():
            project_id = row['project_id']
            short_name = row['project_name']
            s3_folder = f"{short_name}/"
            parquet_key = f"{s3_folder}search_results_baseline.parquet"
            local_parquet = os.path.join(scratch_dir, f"{short_name}_search_results_baseline.parquet")

            try:
                download_s3_file(profile_name, bucket, parquet_key, local_parquet)
            except Exception as e:
                print(f"No baseline S3 results for {short_name} ({project_id}): {e}")
                failed_project_ids.append(project_id)
                continue

            combined.append(_explode_project_parquet(local_parquet, project_id))

            # Sanity check only (prototyping phase) — never blocks the run.
            meta_key = f"{s3_folder}search_results_baseline_meta.json"
            local_meta = os.path.join(scratch_dir, f"{short_name}_meta.json")
            try:
                download_s3_file(profile_name, bucket, meta_key, local_meta)
                with open(local_meta) as f:
                    meta = json.load(f)
                expected_total = meta['summary']['total_images']
                actual_total = sum(len(r) for r in pd.read_parquet(local_parquet)['aoi_stats'])
                if actual_total > expected_total:
                    print(f"WARNING: {short_name} exploded image count ({actual_total}) exceeds "
                          f"the S3 search's own total ({expected_total}) — investigate.")
            except Exception as e:
                print(f"Sanity-check meta.json unavailable for {short_name}: {e}")
        # scratch_dir and every downloaded file in it are deleted automatically here,
        # once the combined CSV below has been saved.

    img_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame(
        columns=['project_id', 'poly_id', 'datetime',
                 'area:cloud_cover_percentage', 'view:sun_elevation', 'area:avg_off_nadir_angle']
    )
    img_df.to_csv(maxar_fp, index=False)
    return failed_project_ids


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
    ev_range = resolve_indicator_window_range(params, 'EARLY_INSIGHT')
    
    proj_df.columns = proj_df.columns.str.lower()

    # Was: img_df = pd.read_csv(maxar_fp, dtype={"datetime": "string"}) reading a
    # manually-copied CSV. Now: build that same combined CSV fresh from S3, then read
    # it back exactly as before — keeps the rest of this function's parsing/windowing
    # logic completely unchanged.
    failed_project_ids = build_maxar_meta_from_s3(params, proj_df, maxar_fp)
    img_df = pd.read_csv(maxar_fp, dtype={"datetime": "string"})
    img_df = img_df[[
            'project_id', 'poly_id',
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

    # Flag projects with no S3 search results yet — distinguishes "not searched" from
    # "searched, genuinely zero images" rather than letting both read as a silent 0.
    if failed_project_ids:
        append_note(final_summary,
                    final_summary['project_id'].isin(failed_project_ids),
                    'no-s3-imagery-data', col='notes_base')

    return final_summary
