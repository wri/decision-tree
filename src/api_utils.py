import requests
import yaml
import json
import pandas as pd
import os
import json
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import rasterio as rs
from rasterio.plot import show
import rasterio.mask
from exactextract import exact_extract
import richdem as rd
import tempfile
import math
import ast
from shapely.geometry import shape
from osgeo import gdal
from rasterio.io import MemoryFile
from tm_api_utils import pull_tm_api_data
import boto3
import os

def upload_to_s3(params, config_path, project_name, today):
    '''
    s3://restoration-monitoring/tree_verification/output/project_data/project_shortname/geojson
    '''
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    aak = config['aws']['aws_access_key_id']
    ask = config['aws']['aws_secret_access_key']
    bucket = params["s3"]["bucket_name"]
    folder_prefix = params["s3"]["folder_prefix"]
    filename = f"{project_name}_{today}.geojson" 
    folder_path = f"{folder_prefix}/{project_name}/geojson/" # UPDATE TO SHORTNAME
    s3_key = f"{folder_path}{filename}"
    local_path = os.path.join(params["outfile"]["geojsons"], filename)
    print(f"local:{local_path}")
    print(f"s3:{s3_key}")
    
    s3 = boto3.client('s3', 
                      aws_access_key_id=aak, 
                      aws_secret_access_key=ask)

    # Check if the project folder exists
    result = s3.list_objects_v2(Bucket=bucket, Prefix=folder_path, MaxKeys=1)
    if 'Contents' not in result:
        # Create folder markers
        s3.put_object(Bucket=bucket, Key=folder_path)

    s3.upload_file(local_path, bucket, s3_key)
    

def patched_pull_tm_api_data(url: str, headers: dict, params: dict) -> list:
    """
    ## REMOVE ME ONCE PACKAGE UPDATED ##
    
    Optimized version of patched_pull_tm_api_data with improved performance.

    - Parses response JSON only once per request
    - Efficiently handles pagination cursors
    - Uses safer access to nested metadata
    """
    results = []
    last_record = None

    while True:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise ValueError(f"Request failed: {response.status_code}")

        response_json = response.json()
        data_items = response_json.get('data', [])

        if not data_items:
            break

        for item in data_items:
            attributes = item.get('attributes', {})
            attributes['poly_id'] = item.get('meta', {}).get('page', {}).get('cursor') or item.get('id')
            results.append(attributes)

        # Efficient cursor handling after processing all records
        new_last_record = data_items[-1].get('meta', {}).get('page', {}).get('cursor') if data_items else None

        if new_last_record and new_last_record != last_record:
            last_record = new_last_record
            params['page[after]'] = last_record
        else:
            break

    return results


def tm_pull_wrapper(params_path, project_ids):
    """
    Wrapper function around the TM API package.

    Iterates over a list of project IDs and aggregates results.
    Saves GeoJSON locally only if geojson_dir is provided.  
    Uploads GeoJSON to S3 only if geojson_s3_bucket is set.
    Skips local file creation entirely if only S3 is used

    Parameters:
        url (str): TerraMatch API endpoint.
        headers (dict): Auth headers.
        project_ids (list): List of project UUIDs.
        outfile (str): Optional path to write output JSON.
        geojson_dir (str): Optional directory to save project-level GeoJSONs.
    """
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    url = params['tm_api']['tm_prod_url']
    out = params['outfile']
    today = out['today']
    outfile = out['tm_response'].format(cohort=out['cohort'], today=out['today'])
    geojson_dir = out['geojsons']
    tm_auth_path = params['tm_api']['tm_auth_path']

    with open(tm_auth_path) as auth_file:
        auth = yaml.safe_load(auth_file)
    headers = {'Authorization': f"Bearer {auth['access_token']}"}
    
    all_results = []
    
    if geojson_dir:
        os.makedirs(geojson_dir, exist_ok=True)

    with tqdm(total=len(project_ids), desc="Pulling Projects", unit="project") as progress:
        for project_id in project_ids:
            api_param_dict = {
                'projectId[]': project_id,
                'polygonStatus[]': 'approved',
                'includeTestProjects': 'false',
                'page[size]': '100'
            }

            try:
                #update this to directly use TM Api 
                results = patched_pull_tm_api_data(url, headers, api_param_dict) 
                if results is None:
                    print(f"No results returned for project: {project_id}")
                    progress.update(1)
                    continue

                # Add project_id for traceability
                for r in results:
                    r['project_id'] = project_id

                all_results.extend(results)
            
            except Exception as e:
                print(f"Error pulling project {project_id}: {e}")
            
            try:
                gdf = gpd.GeoDataFrame(
                    results,
                    geometry=[shape(feature['geometry']) for feature in results],
                    crs='EPSG:4326'
                )
                project_name = next((r.get('projectShortName') for r in results if r.get('projectShortName')), project_id)
                filename = f"{project_name}_{today}.geojson"
                gdf.to_file(os.path.join(geojson_dir, filename), driver='GeoJSON')

                if params['s3']['upload']:
                    upload_to_s3(params, params["config"], project_name, today)

            except Exception as e:
                print(f"Upload error for {project_name}: {e}")

            progress.update(1)

    if outfile:
        with open(outfile, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {outfile}")

    return all_results

def calculate_high_slope_area(slope_raster, polygon, threshold=20):

    '''
    Masks the slope raster with the polygon to determine the
    percentage of polygon area with steep slope.
    If none of the valid pixels are above thresh then the 
    output will be zero

    NaN values indicate no data available for poly
        1. poly doesnt overlap with raster at all
        2. overlapping region is entirely no data
    returns a percentage and status label to identify these cases
    '''
    # Mask the slope raster with the polygon
    out_image, out_transform = rasterio.mask.mask(slope_raster, [polygon], crop=True, nodata=np.nan)
    out_image = out_image[0] 

    # Flatten and analyze the values
    slope_values = out_image.flatten()
    valid_pixels = slope_values[~np.isnan(slope_values)]

    # should only get NaN if all valid values in the raster are NaN
    if len(valid_pixels) == 0:
        return np.nan

    high_slope_pixels = np.sum(valid_pixels > threshold)
    percentage = (high_slope_pixels / len(valid_pixels)) * 100
    return round(percentage, 1)

def opentopo_pull_wrapper(dem_url, api_key, input_df, geojson_dir, slope_thresh, outfile):
    '''
    Downloads DEM data using the project bounding box, then calculates 
    slope and aspect rasters, saved as temp disk files. Calculates zonal statistics 
    for an entire project (due to exact extract specifications) but at the polygon level. 
    If a polygon falls outside the DEM extent, stats default to NaN.

    Parameters:
        dem_url (str): API endpoint for DEM.
        api_key (str): API key for authentication.
        input_df (pd.DataFrame): DataFrame containing list of project_ids to process.
        geojson_dir (str): Directory where per-project GeoJSON files are saved.
        outfile (str): Output CSV file path.
    '''
    project_ids = input_df['project_id'].unique()

    all_prjs = []

    for project_id in tqdm(project_ids, desc='Processing Projects', unit='project'):
        
        # import prj geojson
        geojson_path = os.path.join(geojson_dir, f"{project_id}.geojson")
        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON for {project_id} not found, skipping.")
            continue
        prj = gpd.read_file(geojson_path)
        total_bounds = prj.total_bounds  # (minx, miny, maxx, maxy)

        query = {
            'demtype': 'NASADEM',
            'south': str(total_bounds[1]),
            'north': str(total_bounds[3]),
            'west': str(total_bounds[0]),
            'east': str(total_bounds[2]),
            'outputFormat': 'GTiff',
            'API_Key': api_key
        }

        # Create temporary files for DEM, slope, and aspect
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as dem_tmp, \
             tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as slope_tmp:
            
            dem_path = dem_tmp.name
            slope_path = slope_tmp.name
            
        try:
            # Download the DEM
            response = requests.get(dem_url, stream=True, params=query)
            if response.status_code != 200:
                print(f"Error downloading DEM for project {project_id}: {response.text}")
                raise Exception(f"Failed to download DEM for project {project_id}")

            with open(dem_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            # convert degrees to meters
            latitude = (total_bounds[1] + total_bounds[3]) / 2
            meters_per_degree = 111320 * math.cos(math.radians(latitude))
            zscale = 1 / meters_per_degree

            # Load the DEM and generate slope/aspect
            dem = rd.LoadGDAL(dem_path)
            slope = rd.TerrainAttribute(dem, attrib='slope_percentage', zscale=zscale).astype('float32')

            with rs.open(dem_path) as dem_r:
                profile = dem_r.profile
            profile.update(dtype='float32', count=1)

            with rs.open(slope_path, 'w', **profile) as dst:
                dst.write(slope, 1)
            slope_raster = rs.open(slope_path)

            if slope_raster.crs != prj.crs:
                print("CRS mismatch between raster and gdf!")

            # get zonal stats for the entire prj at once
            aggregations = ['min', 'max', 'mean', 'median', 'majority']
            slope_stats_list = exact_extract(slope_raster, prj, aggregations)

            for idx, (row, slope_zs) in enumerate(zip(prj.itertuples(), slope_stats_list)):
                
                if isinstance(slope_zs, dict) and 'properties' in slope_zs:
                    slope_stats = slope_zs['properties']
                else:
                    slope_stats = {stat: np.nan for stat in aggregations}
                # now calculate the area statistic for the decision
                area_stat, status = calculate_high_slope_area(slope_raster, 
                                                      row.geometry, 
                                                      threshold=slope_thresh)

                all_prjs.append({
                    'project_id': project_id,
                    'poly_id': getattr(row, 'poly_id'),
                    'slope_stats': slope_stats,
                    'slope_area': area_stat,
                })

            slope_raster.close()

        finally:
            # Delete all temp files
            for fpath in [dem_path, slope_path]:
                if os.path.exists(fpath):
                    os.remove(fpath)

    # Write results to CSV
    result_df = pd.DataFrame(all_prjs)
    result_df = result_df.round(1)

    # Expand the nested dictionaries
    result_df = pd.concat([
        result_df.drop(['slope_stats'], axis=1),
        result_df['slope_stats'].apply(pd.Series).add_suffix('_slope'),
    ], axis=1)

    result_df.to_csv(outfile, index=False)
