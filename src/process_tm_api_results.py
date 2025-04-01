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

    Handles issues with dates pertaining to:
    1. Missing dates
    2. Dates that fall before the minimum valid planting date ('1888-01-01')
    2. Illegal month error resulting from NaN values
    3. Start date calculation lands on a leap year
    """
    df[column_name] = df[column_name].astype(str)

    # Replace known invalid date formats with NaT
    df[column_name] = df[column_name].replace(['0000-00-00', 'nan', 'NaN', 'None', '', 'null'], pd.NaT)
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

    # Handle invalid dates that fall before the minimum valid plantstart (Jan 1, 2022) (based on when TF projects began planting)
    min_valid_date = pd.Timestamp("2022-01-01")
    invalid_dates = df[column_name] < min_valid_date
    df.loc[invalid_dates, column_name] = pd.NaT

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
    """
    Identifies where there are missing planting dates for a polygon, hindering Maxar metadata retrieval.
    """
    # Count how many rows are missing start and end dates
    missing_both = df['plantstart'].isna() & df['plantend'].isna()
    print(f"⚠️ Total rows missing start and end plant date: {missing_both.sum()}")

    # Count total polygons per project before filtering
    project_poly_counts = df.groupby('project_id')['poly_id'].nunique() # count of polys per project
    missing_start = df[df['plantstart'].isna()]
    num_projects_missing_start = missing_start['project_id'].nunique()
    num_polygons_missing_start = missing_start['poly_id'].nunique()
    print(f"⚠️ Total projects with at least 1 polygon missing 'plantstart': {num_projects_missing_start}")
    print(f"⚠️ Total polygons missing 'plantstart': {num_polygons_missing_start}")

    # Identify projects where all or partial polygons are missing plantstart
    missing_polys = missing_start.groupby('project_id')['poly_id'].nunique()
    project_poly_counts = project_poly_counts.reindex(missing_polys.index, fill_value=0)

    full_proj_missing = missing_polys[
        missing_polys == project_poly_counts
    ].index.tolist()

    partial_proj_missing = missing_polys[
        missing_polys < project_poly_counts
    ].index.tolist()

    print(f"There are {len(full_proj_missing)} projects with ALL polygons missing plantstart.")
    if len(full_proj_missing) > 0:
        print(f"Projects with ALL polygons missing plantstart: {full_proj_missing}")
    print(f"Projects with SOME polygons missing plantstart: {len(partial_proj_missing)}")

    return df

def process_tm_api_results(results, outfile1, outfile2):
    """
    Processes TerraMatch API results into a clean DataFrame for analysis.
    1. 
    """
    extracted_data = []
    input_ids = {project.get('project_id') for project in results if project.get('project_id')}

    # Iterate over each project to extract project details
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
            'project_phase': project_phase
        })

    final_df = pd.DataFrame(extracted_data)
    final_df.columns = final_df.columns.str.lower()
    pre_clean_ids = list(set(final_df['project_id']))

    # Clean up start and end dates
    final_df = clean_datetime_column(final_df, 'plantstart')
    final_df = clean_datetime_column(final_df, 'plantend')

    # Print out statistics on missing planting dates
    final_df = missing_planting_dates(final_df)

    # Identify missing projects
    output_ids = list(set(final_df['project_id']))
    assert len(input_ids) == len(pre_clean_ids) == len(output_ids)

    missing_projects = input_ids - set(final_df['project_id'])
    if missing_projects:
        print(f"Missing prj ids: {missing_projects}")
    
    # Save the cleaned dataframe to a csv
    final_df.to_csv(outfile1, index=False)
    final_df.to_csv(outfile2, index=False)

    return final_df