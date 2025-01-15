import pandas as pd
import geopandas as gpd
import numpy as np
import os


def imagery_features(imagery_dir, 
                     project_shapefile,
                     cloud_thresh):
    
    '''
    Using the plantstart date from the TM project shapefile
    calculates the number of images available between img date and 
    the plant start date.
    Creates columns to count the number of images available for 
    early verification and baseline assessments.
    '''

    all_proj = gpd.read_file(project_shapefile) # confirm w/ Rhiannon
    imagery_files = os.listdir(imagery_dir)
    subdf_list = []
    for project in imagery_files:
        project_df = pd.read_csv(f"{imagery_dir}/{project}")
        project_sub_df = project_df[['Name', 
                                     'properties.datetime','collection', 
                                     'properties.eo:cloud_cover', 
                                     'properties.off_nadir_avg']]
        project_sub_df['Project'] = project.replace('afr100_', '').replace('_imagery_availability.csv', '')
        subdf_list.append(project_sub_df)

    all_projects_imagery_df = pd.concat(subdf_list).reset_index()
    all_projects_imagery_df = all_projects_imagery_df[['Project', 
                                                       'Name', 
                                                       'properties.datetime',
                                                       'collection', 
                                                       'properties.eo:cloud_cover', 
                                                       'properties.off_nadir_avg']]
    all_projects_imagery_df = all_projects_imagery_df[~pd.isna(all_projects_imagery_df['Name'])]
    all_projects_imagery_df.rename(columns={'Name':'poly_name'}, inplace=True)
    all_projects_imagery_df['properties.datetime'] =pd.to_datetime(all_projects_imagery_df['properties.datetime'], 
                                                                   format='mixed').dt.normalize()
    all_projects_imagery_df['properties.datetime'] = all_projects_imagery_df['properties.datetime'].apply(lambda x: x.replace(tzinfo=None))

    proj_plant_date = all_proj[['Project',  'poly_name', 'plantstart']]
    df = proj_plant_date.merge(all_projects_imagery_df, on=['Project', 'poly_name'], how='left')
    df = df[~pd.isna(df['plantstart'])]
    df = df[df['plantstart'] != '15-04-2024 - 15-05-2024']
    df['plantstart'] = pd.to_datetime(df['plantstart'], format='mixed')
    df['properties.datetime'] = pd.to_datetime(df['properties.datetime'], format='mixed').dt.normalize()
    df['properties.datetime'] = df['properties.datetime'].apply(lambda x: x.replace(tzinfo=None))
    df['dateDiff']= (df['properties.datetime'] - df['plantstart']).dt.days

    # get baseline img count
    df_imagery_usable_baseline = df[(df['dateDiff'] < 0 ) & (df['dateDiff'] > -365 ) & (df['properties.eo:cloud_cover'] < cloud_thresh)]
    usable_baseline_summary = df_imagery_usable_baseline.groupby(['Project', 'poly_name']).count().reset_index()[['Project', 
                                                                                                                'poly_name', 
                                                                                                                'collection']]
    usable_baseline_summary.rename(columns={'collection':'available_baseline_images'}, inplace=True)

    # get early verf img count
    df_imagery_usable_ev = df[(df['dateDiff'] > 90 ) & (df['properties.eo:cloud_cover'] < cloud_thresh)]
    usable_ev_summary = df_imagery_usable_ev.groupby(['Project', 'poly_name']).count().reset_index()[['Project', 
                                                                                                    'poly_name', 
                                                                                                    'collection']]
    usable_ev_summary.rename(columns={'collection':'available_ev_images'}, inplace=True)

    # combine
    all_proj_df_imagery = all_proj.merge(usable_baseline_summary , on=['Project', 'poly_name'], how='left')
    all_proj_df_imagery = all_proj_df_imagery.merge(usable_ev_summary, on=['Project', 'poly_name'], how='left')

    all_proj_df_imagery['available_baseline_images'] = all_proj_df_imagery['available_baseline_images'].replace({np.nan:int(0)})
    all_proj_df_imagery['available_ev_images'] = all_proj_df_imagery['available_ev_images'].replace({np.nan:int(0)})

    return all_proj_df_imagery



# ## should also perform this step
#     ## BRANCH 3 ##
#     def image_availability(row):
#         method = 'field' if row.baseline_img < 1 else 'remote'
#         return method
#     open_sys = open_sys.assign(method=open_sys.apply(image_availability, axis=1))
#     closed_sys['method'] = 'field'