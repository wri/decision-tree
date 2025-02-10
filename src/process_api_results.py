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

    # handle leap year instances
    is_feb_29 = df[column_name].notna() & (df[column_name].dt.month == 2) & (df[column_name].dt.day == 29)
    years = df.loc[is_feb_29, column_name].dt.year
    non_leap_years = ~years.apply(calendar.isleap)
    df.loc[is_feb_29 & non_leap_years, column_name] = df.loc[is_feb_29 & non_leap_years, column_name].apply(
        lambda x: datetime(x.year, 2, 28)
    )
    missing_count = df[column_name].isna().sum()
    print(f"Number of rows missing a '{column_name}' date: {missing_count}/{len(df)}")
    return df
    

def process_tm_api_results(results, outfile1, outfile2):
    """
    Processes API results into a clean DataFrame for analysis.
    """
    project_df = pd.DataFrame(results)

    tree_cover_data = []

    # Iterate over each project to extract tree cover statistics
    for project in results:
        project_id = project.get('project_id')
        poly_id = project.get('poly_id')
        site_id = project.get('siteId')
        indicators = project.get('indicators', [])

        # Iterate through indicators to find tree cover
        for indicator in indicators:
            if indicator.get('indicatorSlug') == 'treeCover':
                year = indicator.get('yearOfAnalysis')
                percent_cover = indicator.get('percentCover')

                tree_cover_data.append({
                    'project_id': project_id,
                    'poly_id': poly_id,
                    'siteId': site_id,
                    f'ttc_{year}': percent_cover 
                })

    # Merge tree cover -- if there is no tree cover val it will be NaN
    tree_cover_df = pd.DataFrame(tree_cover_data)
    mrgd = project_df.merge(tree_cover_df, on=['project_id', 'poly_id', 'siteId'], how='left') 

    # clean and drop
    mrgd.columns = mrgd.columns.str.lower()
    drop_columns = ['status', 'establishmenttreespecies', 'reportingperiods', 'indicators']
    mrgd = mrgd.drop(drop_columns, axis=1)

    # clean up start and end dates
    final_df = clean_datetime_column(mrgd, 'plantstart')
    final_df = clean_datetime_column(mrgd, 'plantend')
    missing_dates_count = final_df['plantstart'].isna() & final_df['plantend'].isna()
    total_missing = missing_dates_count.sum()
    print(f"Total rows missing start and end plant date: {total_missing}")
    final_df = final_df.dropna(subset=['plantstart'], how='any')
    
    final_df.to_csv(outfile1, index=False)
    final_df.to_csv(outfile2, index=False)
    
    # if outfile is not None:
    #     final_df.to_csv(outfile)

    return final_df