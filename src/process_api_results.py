import pandas as pd
import json
import calendar
from datetime import datetime

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
    affected_rows = is_feb_29.copy()
    affected_rows.loc[is_feb_29] = non_leap_years

    df.loc[affected_rows, column_name] = df.loc[affected_rows, column_name].apply(
        lambda x: datetime(x.year, 2, 28)
    )

    missing_count = df[column_name].isna().sum()
    print(f"Number of rows missing a '{column_name}' date: {missing_count}/{len(df)}")
    return df


def missing_planting_dates(df):
    '''
    Identifies where there are missing planting dates for 
    a polygon, hindering maxar metadata retrieval
    '''
    # count how rows are missing start and end dates
    missing_both = df['plantstart'].isna() & df['plantend'].isna()
    print(f"⚠️ Total rows missing start and end plant date: {missing_both.sum()}")

    # Count total polygons per project before filtering
    project_poly_counts = df.groupby('project_id')['poly_id'].nunique() # count of polys per prj
    missing_start = df[df['plantstart'].isna()]
    num_projects_dropped = missing_start['project_id'].nunique()
    num_polygons_dropped = missing_start['poly_id'].nunique()
    print(f"⚠️ Total projects missing 'plantstart': {num_projects_dropped}")
    print(f"⚠️ Total polygons missing 'plantstart': {num_polygons_dropped}")
     
    # Identify projects where all or partial polys are removed
    dropped_polys = missing_start.groupby('project_id')['poly_id'].nunique()
    project_poly_counts = project_poly_counts.reindex(dropped_polys.index, fill_value=0)

    full_proj_drops = dropped_polys[
        dropped_polys == project_poly_counts
    ].index.tolist()

    partial_proj_drops = dropped_polys[
        dropped_polys < project_poly_counts
    ].index.tolist()

    print(f"Projects fully removed: {len(full_proj_drops)}")
    print(f"Projects partially affected: {len(partial_proj_drops)}")
    # drop row if plantstart has missing vals (Nan or None)
    final_df = df.dropna(subset=['plantstart'], how='any')
    return final_df


def process_tm_api_results(results, outfile1, outfile2):
    """
    Processes API results into a clean DataFrame for analysis.

    Extracting tree cover indicators:
    **tree_cover_years syntax is a Python dictionary unpacking 
    feature that allows us to dynamically add multiple columns 
    to the DataFrame without explicitly defining them.
    """
    extracted_data = []
    input_ids = {project.get('project_id') for project in results if project.get('project_id')}

    # Iterate over each project to extract project details & tree cover statistics
    for project in results:
        # Extract general project details
        project_id = project.get('project_id')
        poly_id = project.get('poly_id')
        site_id = project.get('siteId')
        geom = project.get('geometry')
        plant_start = project.get('plantStart')
        plant_end = project.get('plantEnd')
        practice = project.get('practice')
        targetsys = project.get('targetSys')
        dist = project.get('distr')
        project_phase = project.get('projectPhase', '')  # Ensure a default empty string

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
            'geometry': geom,
            'plantstart': plant_start,
            'plantend': plant_end,
            'practice': practice,
            'target_sys': targetsys,
            'dist': dist,
            'project_phase': project_phase,
            **tree_cover_years  
        })

    final_df = pd.DataFrame(extracted_data)
    final_df.columns = final_df.columns.str.lower()
    pre_clean_ids = list(set(final_df['project_id']))

    # Clean up start and end dates
    final_df = clean_datetime_column(final_df, 'plantstart')
    final_df = clean_datetime_column(final_df, 'plantend')

    # Drop polygons with missing planting dates
    final_df = missing_planting_dates(final_df)

    # identify missing projs
    output_ids = list(set(final_df['project_id']))
    assert len(input_ids) == len(pre_clean_ids) == len(output_ids)

    missing_projects = input_ids - set(final_df['project_id'])
    if missing_projects:
        print(f"Missing prj ids: {missing_projects}")

    final_df.to_csv(outfile1, index=False)
    final_df.to_csv(outfile2, index=False)

    # if outfile is not None:
    #     final_df.to_csv(outfile)

    return final_df