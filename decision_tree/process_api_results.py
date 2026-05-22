import pandas as pd
import ast
import calendar
from datetime import datetime
import geopandas as gpd
import os
from shapely import wkb

def process_tm_results(params, 
                       results: str, 
                       geojson_dir: str, 
                       project_ids=None, 
                       limit_to_test_projects: bool = False, 
                       save_geojsons: bool = True):
    """
    Read GeoParquet file, flatten it into a tabular dataframe,
    run cleaning steps, and optionally save project-level GeoJSONs.

    Returns: Cleaned dataframe for downstream decision-tree analysis.
    """
    out = params.get("outfile", {})
    criteria = params.get("criteria", {})
    drop_missing = criteria.get("drop_missing", False)
    cohort_raw = out['cohort']
    cohort = 'terrafund-cohort-1' if cohort_raw == 'c1' else 'terrafund-cohort-2'

    # Ingest polygon data
    df = _read_geoparquet(results)

    # Filter for test projects
    test_short_name_prefix = 'TEST_'
    if limit_to_test_projects:
        filtered_df = df[df['short_name']
                         .str.contains(test_short_name_prefix, case=False, na=False)].reset_index(drop=True)
    else:
        filtered_df = df[~df['short_name']
                         .str.contains(test_short_name_prefix, case=False, na=False) 
                         | df['short_name'].isna()].reset_index(drop=True)

    # Filter to specified projects for projectid mode
    if project_ids:
        filtered_df = filtered_df[filtered_df['project_id']
                                  .isin(project_ids)].reset_index(drop=True)

    # Filter and rename for standard columns
    raw_df = flatten_tm_geoparquet(filtered_df)

    # Filter to specified cohort
    raw_df = raw_df[raw_df["cohort"].str.contains(cohort, na=False)]

    raw_df.columns = raw_df.columns.str.lower()
    input_ids = set(raw_df["project_id"].dropna().unique())
    pre_clean_ids = set(raw_df["project_id"].dropna().unique())
    raw_df['notes'] = None

    clean_df = clean_datetime_column(raw_df, "plantstart")
    clean_df = missing_planting_dates(clean_df, drop_missing)
    clean_df = missing_features(clean_df, drop_missing)
    clean_df["practice"] = clean_df["practice"].apply(normalize_practice)
    clean_df = resolve_multipractice(clean_df)
    if save_geojsons:
        data_version = out.get("data_version")
        save_project_geojsons(clean_df, geojson_dir, data_version)

    output_ids = set(clean_df["project_id"].dropna().unique())
    poly_ids = set(clean_df["poly_id"].dropna().unique())
    total_area = sum(clean_df["area"])

    assert len(input_ids) == len(pre_clean_ids) == len(output_ids), (
        f"input: {len(input_ids)}, "
        f"preclean: {len(pre_clean_ids)}, "
        f"output: {len(output_ids)}"
    )
    print(
        f"\nRunning forecast for {cohort} cohort\n"
        f"{cohort} has a total of:\n"
        f"  {len(output_ids)} total projects\n"
        f"  {len(poly_ids)} total polygons\n" 
        f"  {total_area:,.2f} total hectares" 
    )
    return clean_df


def _read_geoparquet(results_path):
    """
    Read parquet with pandas and standardize column names.
    """
    df = pd.read_parquet(results_path)
    df.columns = df.columns.str.lower()
    return df


def flatten_tm_geoparquet(results):
    """
    Flatten GeoParquet rows into the tabular schema expected by the pipeline.

    This function:
    - extracts key project and polygon attributes
    - parses the cohort field from parquet bytes into a Python list
    - optionally filters rows to a requested cohort
    - expands the `ttc` field into year-specific tree cover columns
    """
    records = []

    for row in results.itertuples(index=False):
        row_dict = row._asdict()
        
        # parse geometry
        geom_bytes = row_dict.get("geom")
        try:
            geometry = wkb.loads(geom_bytes) if geom_bytes is not None else None
        except Exception:
            geometry = None

        cohort_val = row_dict.get("cohort")
        cohort = cohort_val.decode("utf-8").strip('[]').strip('"') if cohort_val else None

        tree_cover_years = extract_tree_cover_years(row_dict)

        records.append({
            "cohort": cohort,
            "project_id": row_dict.get("project_id"),
            "poly_id": row_dict.get("poly_uuid"),
            "site_id": row_dict.get("site_id"),
            "project_name": row_dict.get("short_name"),
            "geometry": geometry,
            "plantstart": row_dict.get("plantstart"),
            "practice": row_dict.get("practice"),
            "target_sys": row_dict.get("target_sys"),
            "area": row_dict.get("calc_area"),
            **tree_cover_years,
        })

    return pd.DataFrame(records)


def extract_tree_cover_years(row_dict):
    """
    Extract tree cover values from the `ttc` field and convert them into
    flat columns like `ttc_2021`, `ttc_2022`, etc.
    row_dict["ttc"] is a list of tuples, where first value is the year
    and second value is the tree cover percent
    """
    out = {}
    ttc_values = row_dict.get("ttc", [])
    if not isinstance(ttc_values, list):
        return out

    current_year = datetime.today().year
    for item in ttc_values:
        year, percent_cover = item
        year = int(year)
        if 2020 <= year <= current_year:
            out[f"ttc_{year}"] = percent_cover

    return out

def save_project_geojsons(df, geojson_dir, data_version):
    """
    Save one GeoJSON per project.
 
    Assumes upstream cleaning has already handled dates, missing values,
    and practice normalization. This function also:
    - Flags and skips null or empty geometries
    - Attempts to repair invalid geometries with buffer(0); skips if unrecoverable
    - Serializes list-type fields (e.g. dist) that are unsupported by OGR (type 5 = RealList)
    Problematic polygons persist but are not written to disk.
    """
    os.makedirs(geojson_dir, exist_ok=True)
    print("Saving project GeoJSONs...")
 
    for project_id, project_df in df.groupby("project_id", dropna=False):
        project_names = project_df["project_name"].dropna().unique()
        if len(project_names) == 0:
            print(f"{project_id} has no project_name")
            continue
        project_name = project_names[0]
        valid_rows = []
        skipped = []
 
        for row in project_df.itertuples(index=False):
            poly_id = getattr(row, "poly_id", None)
            row_dict = dict(row._asdict())
            geometry = row_dict.get("geometry")
 
            # --- Geometry checks ---
            if geometry is None:
                skipped.append((poly_id, "null geometry"))
                continue
 
            if geometry.is_empty:
                skipped.append((poly_id, "empty geometry"))
                continue
 
            if not geometry.is_valid:
                fixed = geometry.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    print(
                        f"⚠️ Repaired invalid geometry: "
                        f"project={project_name}, poly_id={poly_id}"
                    )
                    row_dict["geometry"] = fixed
                else:
                    skipped.append((poly_id, "invalid geometry (unrecoverable)"))
                    continue
 
            # --- Serialize list-type fields unsupported by OGR (type 5 = RealList) ---
            # Affects fields like 'dist' which may be stored as float lists in the parquet
            for field, val in row_dict.items():
                if field == "geometry":
                    continue
                if isinstance(val, (list, tuple)):
                    row_dict[field] = ",".join(str(v) for v in val)
 
            valid_rows.append(row_dict)
 
        if skipped:
            for poly_id, reason in skipped:
                print(
                    f"⚠️ Skipping polygon: project={project_name}, "
                    f"project_id={project_id}, poly_id={poly_id}, reason={reason}"
                )
                df.loc[df['poly_id'] == poly_id, 'notes'] = 'invalid-geometry'
 
        if not valid_rows:
            print(f"⚠️ No valid polygons for {project_name} ({project_id}); skipping GeoJSON.")
            continue
 
        out_gdf = gpd.GeoDataFrame(
            valid_rows,
            geometry="geometry",
            crs="EPSG:4326",
        )
 
        # GeoJSON writing can fail on pandas Timestamp fields, so cast datetimes to string
        for col in out_gdf.columns:
            if pd.api.types.is_datetime64_any_dtype(out_gdf[col]):
                out_gdf[col] = out_gdf[col].astype(str)
 
        filename = f"{project_name}_{data_version}.geojson"
        outpath = os.path.join(geojson_dir, filename)
        out_gdf.to_file(outpath, driver="GeoJSON")
 
    return None

def clean_datetime_column(df, column_name):
    """
    Cleans a datetime column in a pandas DataFrame by:
    1. Replacing invalid date strings ('0000-00-00') with NaT.
    2. Converting the column to datetime format with error handling.
    3. Printing the number of missing dates after conversion.

    Handles issues with dates pertaining to 1) missing dates
    2) illegal month error resulting from NaN values 3) start
    date calculation lands on a leap year
    """
    df[column_name] = df[column_name].astype(str)

    # Replace known invalid date formats with NaT
    df[column_name] = df[column_name].replace(['0000-00-00', 'nan', 'NaN', 'None', '', 'null'], pd.NaT)
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

    # Handle leap year Feb 29 cases
    is_feb_29 = df[column_name].notna() & (df[column_name].dt.month == 2) & (df[column_name].dt.day == 29)
    years = df.loc[is_feb_29, column_name].dt.year

    # Ensure aligned index with is_feb_29
    non_leap_years = ~years.apply(calendar.isleap)
    affected_rows = is_feb_29.where(~is_feb_29, non_leap_years.values)

    df.loc[affected_rows, column_name] = df.loc[affected_rows, column_name].apply(
        lambda x: datetime(x.year, 2, 28)
    )
    return df

def missing_planting_dates(df, drop=False):
    '''
    Identifies where there are missing planting dates for 
    a polygon, hindering maxar metadata retrieval
    Option to drop rows with missing dates
    '''

    # Count total polygons per project before filtering
    project_poly_counts = df.groupby('project_id')['poly_id'].nunique() # count of polys per prj
    missing_start = df[df['plantstart'].isna()]
    df.loc[missing_start.index, 'notes'] = 'missing-plantstart'

    # invalid plantstart
    current_year = datetime.today().year
    plantstart_year = pd.to_datetime(df['plantstart'], errors='coerce').dt.year
    invalid_year_mask = (plantstart_year < 2020) | (plantstart_year > current_year)
    df.loc[invalid_year_mask, 'notes'] = 'invalid-plantstart'

    for _, row in missing_start[['project_name', 'project_id', 'poly_id']].drop_duplicates().iterrows():
        print(f"{row['project_name']} | {row['project_id']} | {row['poly_id']}")

    num_projects_dropped = missing_start['project_id'].nunique()
    num_polygons_dropped = missing_start['poly_id'].nunique()
    print(f"⚠️ Projects missing 'plantstart': {num_projects_dropped}")
    print(f"⚠️ Polygons missing 'plantstart': {num_polygons_dropped}/{len(df)}")
     
    # Identify projects where all or partial polys are removed
    dropped_polys = missing_start.groupby('project_id')['poly_id'].nunique()
    project_poly_counts = project_poly_counts.reindex(dropped_polys.index, fill_value=0)

    full_proj_drops = dropped_polys[
        dropped_polys == project_poly_counts
    ].index.tolist()

    partial_proj_drops = dropped_polys[
        dropped_polys < project_poly_counts
    ].index.tolist()

    print(f"Projects fully affected: {len(full_proj_drops)}")
    print(f"Projects partially affected: {len(partial_proj_drops)}")

    # drop row if plantstart has missing vals (Nan or None)
    final_df = df.dropna(subset=['plantstart'], how='any') if drop else df

    return final_df

def missing_features(df, drop=False, save_missing=True):
    '''
    Identifies rows where ttc is only NaN values (no data for any years)
    but only for polygons with plantstart year between 2017 and 2024 inclusive.
    Identifies rows where practice or targetsys is NaN.
    Optionally drops these rows based on the drop argument 
    and prints a statement about the count of rows affected.

    ** assumption: should have a tree cover stat for all approved polygons 
    on TM that started planting before 2024
    '''
    starting = len(df)
    ttc_cols = [col for col in df.columns if col.startswith('ttc_') and col[4:].isdigit()]
    
    plantstart_year = pd.to_datetime(df['plantstart'], errors='coerce').dt.year

    # Only consider missing TTC for plantstart years 2017-2025 inclusive
    eligible_ttc_mask = plantstart_year.between(2017, 2025, inclusive='both')
    null_rows = df[eligible_ttc_mask & df[ttc_cols].isna().all(axis=1)]
    # placeholder for notes label missing-ttc 
    missing_practice = df[df['practice'].isna()]
    df.loc[missing_practice.index, 'notes'] = 'missing-practice'
    missing_targetsys = df[df['target_sys'].isna()]
    df.loc[missing_targetsys.index, 'notes'] = 'missing-target-sys'
    if save_missing and not null_rows.empty:
        print("TTC NaNs saved to file.")
        null_rows.to_csv('ttc_nans.csv', index=False)

    print(f"⚠️ Polygons missing 'ttc': {len(null_rows)}")
    print(f"⚠️ Polygons missing 'practice': {len(missing_practice)}")
    print(f"⚠️ Polygons missing 'target system': {len(missing_targetsys)}")

    # Combine indices for all three conditions
    rows_to_drop = pd.Index(null_rows.index).union(missing_practice.index).union(missing_targetsys.index)

    final_df = df.drop(index=rows_to_drop) if drop else df
    finishing = len(final_df)
    diff = starting - finishing
    print(f"\n{round((diff/starting)*100)}% data lost due to missing values.\n")

    return final_df
    
def resolve_multipractice(df):
    """
    Resolve projects where one row has multiple practices and the other rows
    in the same project consistently use a single practice.

    This function assumes:
    - single-practice rows contain scalar strings like 'tree-planting'
    - multi-practice rows contain a delimiter like '|', for example
      'direct-seeding|tree-planting'

    A multi-practice row is updated only when:
    - the project has exactly one multi-practice row
    - all other rows in that project share the same single practice
    """
    df = df.copy()
    df['is_multipractice'] = df['practice'].astype(str).str.contains(r'\|', na=False)

    # If there are no multi-practice rows, return unchanged
    if not df['is_multipractice'].any():
        print("No multi-practice rows found; nothing to resolve.")
        df.drop(columns='is_multipractice', inplace=True)
        return df

    updated_projects = []

    for project_id, group in df.groupby('project_id'):
        multipractice_rows = group[group['is_multipractice']]
        singlepractice_rows = group[~group['is_multipractice']]

        # Only resolve projects with one multi-practice row and 
        # at least one single-practice row
        if len(multipractice_rows) == 1 and len(singlepractice_rows) > 0:
            unique_single = singlepractice_rows['practice'].dropna().unique()

            # Only update if all single-practice rows agree on one value
            if len(unique_single) == 1:
                consistent_value = unique_single[0]
                idx_to_update = multipractice_rows.index[0]
                df.at[idx_to_update, 'practice'] = consistent_value

                project_name = group['project_name'].iloc[0]
                updated_projects.append((project_name, consistent_value))

    df.drop(columns='is_multipractice', inplace=True)

    if updated_projects:
        for name, val in updated_projects:
            print(f"{name} updated one polygon to -> '{val}'")
    else:
        print("No single-row multi-practice projects were updated.")

    still_multi = df['practice'].astype(str).str.contains(r'\|', na=False)
    df.loc[still_multi, 'notes'] = 'multi-practice'

    return df

def normalize_practice(value):
    """
    Normalize practice values so they match scalar values in the rule template.

    Handles:
    - real Python lists/tuples
    - strings that look like lists, e.g. "['tree-planting']"
    - strings that look like tuples, e.g. "('tree-planting')"
    - plain strings, e.g. "tree-planting"

    Returns:
    - a scalar string for single-practice cases
    - a pipe-delimited string for multi-practice cases
    - pd.NA for missing values
    """
    # Case 1: already a real Python list or tuple
    if isinstance(value, (list, tuple)):
        cleaned = [str(v).strip().lower() for v in value if pd.notna(v)]
        if len(cleaned) == 1:
            return cleaned[0]
        return "|".join(cleaned)

    # Case 2: string input
    if isinstance(value, str):
        s = value.strip()

        # Try to parse strings that look like Python containers
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)

                # If parsed into a real list/tuple, clean it
                if isinstance(parsed, (list, tuple)):
                    cleaned = [str(v).strip().lower() for v in parsed if pd.notna(v)]
                    if len(cleaned) == 1:
                        return cleaned[0]
                    return "|".join(cleaned)

                # If parsed into a scalar, return it
                return str(parsed).strip().lower()

            except (ValueError, SyntaxError):
                pass

        # Otherwise treat it as a plain scalar string
        return s.lower()

    # Fallback for unexpected types
    return str(value).strip().lower()