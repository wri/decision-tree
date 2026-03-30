import pandas as pd
import ast
import calendar
from datetime import datetime
import geopandas as gpd
from shapely.validation import make_valid
import os
import json
from shapely import wkb

def process_tm_results(params, results, geojson_dir, save_geojsons=True):
    """
    Read GeoParquet file, flatten it into a tabular dataframe,
    run cleaning steps, and optionally save project-level GeoJSONs.

    Returns: Cleaned dataframe for downstream decision-tree analysis.
    """
    out = params.get("outfile", {})
    criteria = params.get("criteria", {})
    drop_missing = criteria.get("drop_missing", False)
    cohort_raw = out['cohort']
    cohort = 'terrafund' if cohort_raw == 'c1' else 'terrafund-landscapes'

    df = _read_geoparquet(results)
    raw_df = flatten_tm_geoparquet(df)
    raw_df = raw_df[(raw_df.cohort == cohort)]

    raw_df.columns = raw_df.columns.str.lower()
    input_ids = set(raw_df["project_id"].dropna().unique())
    pre_clean_ids = set(raw_df["project_id"].dropna().unique())

    clean_df = clean_datetime_column(raw_df, "plantstart")
    clean_df = missing_planting_dates(clean_df, drop_missing)
    clean_df = missing_features(clean_df, drop_missing)
    clean_df["practice"] = clean_df["practice"].apply(normalize_practice)
    clean_df = resolve_multipractice(clean_df)
    if save_geojsons:
        data_version = out.get("data_version")
        save_project_geojsons(clean_df, geojson_dir, data_version)

    output_ids = set(clean_df["project_id"].dropna().unique())

    assert len(input_ids) == len(pre_clean_ids) == len(output_ids), (
        f"input: {len(input_ids)}, "
        f"preclean: {len(pre_clean_ids)}, "
        f"output: {len(output_ids)}"
    )
    missing_projects = input_ids - output_ids
    if missing_projects:
        print(f"Missing prj ids: {missing_projects}")

    print(f"Running forecast for {len(output_ids)} projects in {cohort} cohort.")
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
            "poly_id": row_dict.get("poly_id"),
            "site_id": row_dict.get("site_id"),
            "project_name": row_dict.get("project_name"),
            "geometry": geometry,
            "plantstart": row_dict.get("plantstart"),
            "practice": row_dict.get("practice"),
            "target_sys": row_dict.get("target_sys"),
            "dist": row_dict.get("distr"),
            "project_phase": row_dict.get("project_phase"),
            "area": row_dict.get("calcarea"),
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

    for item in ttc_values:
        year, percent_cover = item
        year = int(year)
        out[f"ttc_{year}"] = percent_cover

    return out

def save_project_geojsons(df, geojson_dir, data_version):
    """
    Save one GeoJSON per project.

    Assumes upstream cleaning has already handled dates, missing values,
    and practice normalization. This function only checks whether each
    row's geometry is usable for GeoJSON output.
    """
    os.makedirs(geojson_dir, exist_ok=True)
    print("Saving project GeoJSONs...")

    for project_id, project_df in df.groupby("project_id", dropna=False):
        project_names = project_df["project_name"].dropna().unique()
        project_name = project_names[0] if len(project_names) > 0 else str(project_id)

        valid_rows = []

        for row in project_df.itertuples(index=False):
            poly_id = getattr(row, "poly_id", None)
            row_dict = row._asdict()
            try:
                geometry = row_dict.get("geometry")
                row_dict["geometry"] = geometry
                valid_rows.append(row_dict)
            except Exception as e:
                print(
                    f"Skipping polygon during GeoJSON creation: "
                    f"project_name={project_name}, "
                    f"project_id={project_id}, "
                    f"polygon_id={poly_id}, "
                    f"error={e}"
                )
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

        filename = f"{project_id}_{data_version}.geojson"
        outpath = os.path.join(geojson_dir, filename)
        out_gdf.to_file(outpath, driver="GeoJSON")
        print(f"Saved to {outpath}")

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
    missing_practice = df[df['practice'].isna()]
    missing_targetsys = df[df['target_sys'].isna()]
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

def OLDprocess_tm_results(params, results):
    """
    Filter geoparquet into clean DataFrame for analysis.

    outfile: filepath to store cleaned results
    outfile2: filepath to store cleaned results for maxar metadata request

    Extracting tree cover indicators:
    **tree_cover_years syntax is a Python dictionary unpacking 
    feature that allows us to dynamically add multiple columns 
    to the DataFrame without explicitly defining them.
    """
    # this fails because of a geometry issue - IllegalArgumentException: 
    # Invalid number of points in LinearRing found 2 - must be 0 or >= 3
    gdf = gpd.read_parquet(results) 

    drop_missing = params['criteria']['drop_missing']
    extracted_data = []
    input_ids = {project.get('projectId') for project in results if project.get('projectId')}

    # iterate over fields in the geoparquet and rename them
    for project in results:
        project_id = project.get('projectId')
        poly_id = project.get('poly_id')
        site_id = project.get('siteId')
        project_name = project.get('projectShortName')
        geom = project.get('geometry')
        plant_start = project.get('plantStart')
        practice = project.get('practice')
        targetsys = project.get('targetSys')
        dist = project.get('distr')
        project_phase = project.get('projectPhase', '')  # Ensure a default empty string
        area = project.get('calcArea')

        # Extract tree cover indicators
        tree_cover_years = {}
        indicators = project.get('indicators', [])

        for indicator in indicators:
            try:
                year = int(indicator.get('yearOfAnalysis'))
                if 2017 <= year <= 2024:
                    tree_cover_years[f'ttc_{year}'] = indicator.get('percentCover')
            except (TypeError, ValueError):
                # Skip invalid or missing year values
                continue

        # Store extracted project details in a structured format
        extracted_data.append({
            'project_id': project_id,
            'poly_id': poly_id,
            'site_id': site_id,
            'project_name': project_name,
            'geometry': geom,
            'plantstart': plant_start,
            'practice': practice,
            'target_sys': targetsys,
            'dist': dist,
            'project_phase': project_phase,
            'area': area,
            **tree_cover_years  
        })

    raw_df = pd.DataFrame(extracted_data)
    raw_df.columns = raw_df.columns.str.lower()
    pre_clean_ids = list(set(raw_df['project_id']))

    # Clean up start and end dates, missing info, multipractice issues
    clean_df = clean_datetime_column(raw_df, 'plantstart')
    clean_df = missing_planting_dates(clean_df, drop_missing)
    clean_df = missing_features(clean_df, drop_missing)
    clean_df['practice'] = clean_df['practice'].apply(normalize_practice)
    clean_df = resolve_multipractice(clean_df)

    # final clean up
    output_ids = list(set(clean_df['project_id']))
    assert len(input_ids) == len(pre_clean_ids) == len(output_ids), (
        f"input: {len(input_ids)}, "
        f"preclean: {len(pre_clean_ids)}, "
        f"output: {len(output_ids)}"
    )

    missing_projects = input_ids - set(clean_df['project_id'])
    if missing_projects:
        print(f"Missing prj ids: {missing_projects}")

    return clean_df