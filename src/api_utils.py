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
import rasterio.mask
from exactextract import exact_extract
import richdem as rd
import tempfile
import math
from shapely.geometry import shape
from osgeo import gdal
from tm_api_utils import pull_tm_api_data
from s3_utils import upload_to_s3   
import time
import sys


def get_ids(params):

    print("Requesting project IDs from TerraMatch...")
    out = params['outfile']
    cohort = out['cohort']
    keyword = 'terrafund' if cohort == 'c1' else 'terrafund-landscapes'
    url = params['tm_api']['tm_prod_url']
    out = params['outfile']
    tm_auth_path = params['config']
    
    with open(tm_auth_path) as auth_file:
        auth = yaml.safe_load(auth_file)
    headers = {'Authorization': f"Bearer {auth['access_token']}"}
    api_param_dict = {'projectCohort[]': keyword,
                    'polygonStatus[]': 'approved',
                    'includeTestProjects': 'false',
                    'page[size]': '100'
            }
    try:
        results = pull_tm_api_data(url, headers, api_param_dict, normalize_column_names=False)
        
        ids = list({
        str(r["projectId"])
        for r in results
        if "projectId" in r and r["projectId"] is not None
         })
        if len(ids) < 1:
            print(f"Error: only found {len(ids)} project IDs. Exiting.")
            sys.exit(1)
        else:
            print(f"Found {len(ids)} project IDs for {cohort}.")
    
    except Exception as e:
        print(f"Error pulling project ids: {e}")
    
    return ids 

def get_tm_feats(params, project_ids):
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
    url = params['tm_api']['tm_prod_url']
    out = params['outfile']
    data_version = out['data_version']
    outfile = out['tm_response'].format(cohort=out['cohort'], data_version=data_version)
    geojson_dir = out['geojsons']

    headers = get_gfw_access_token(params)
    
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
                results = pull_tm_api_data(url, headers, api_param_dict, normalize_column_names=False) 
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
                filename = f"{project_name}_{data_version}.geojson"
                gdf.to_file(os.path.join(geojson_dir, filename), driver='GeoJSON')
            except Exception as e:
                print(f"Geojson creation error for {project_name}: {e}")

            if params['s3']['upload']:
                upload_to_s3(params, params["config"], project_name, data_version)

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

def opentopo_pull_wrapper(params, config, feats_df):
    '''
    Checks for existing outputs to reduce API requests.
    Downloads DEM data using the project bounding box + buffer, then calculates 
    slope and aspect rasters, saved as temp disk files. Calculates zonal statistics 
    for an entire project (due to exact extract specifications) at the polygon level. 
    Calculates area with high slope.
    If a polygon falls outside the DEM extent, stats default to NaN.

    Parameters:
        dem_url (str): API endpoint for DEM.
        api_key (str): API key for authentication.
        input_df (pd.DataFrame): DataFrame containing list of project_ids to process.
        geojson_dir (str): Directory where per-project GeoJSON files are saved.
        outfile (str): Output CSV file path.
    '''
    dem_url = params['opt_api']['opt_url']
    api_key = config['opentopo_key']
    geojson_dir = params['outfile']['geojsons']
    slope_thresh = params['criteria']['slope_thresh']
    data_version = params['outfile']['data_version']

    project_names = feats_df['project_name'].unique()
    #project_names = [i for i in project_names if i != 'MLI_22_ASIC'] # this should only drop 1 prj and 100 polys
    dfs_to_concat = []
    for name in project_names:
        project_df = feats_df[feats_df.project_name == name]
        project_id = project_df.project_id.iloc[0]
        stat_path = params['outfile']['project_stats'].format(name=name)
        if os.path.exists(stat_path):
            dfs_to_concat.append(pd.read_csv(stat_path))
            print(f"{name} already processed, skipping.")
            continue
        else:
            print(f"Processing {name}")
        
        geojson_path = os.path.join(geojson_dir, f"{name}_{data_version}.geojson")
        prj = gpd.read_file(geojson_path)
        total_bounds = prj.total_bounds  
        buffer_deg = 0.01  # ~1 km buffer; adjust if needed
        minx, miny, maxx, maxy = total_bounds
        query = {
            'demtype': 'NASADEM',
            'south': str(miny - buffer_deg),
            'north': str(maxy + buffer_deg),
            'west': str(minx - buffer_deg),
            'east': str(maxx + buffer_deg),
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
                raise Exception(f"Error downloading DEM for project {name}: {response.text}")

            with open(dem_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            # convert degrees to meters
            latitude = (total_bounds[1] + total_bounds[3]) / 2
            meters_per_degree = 111320 * math.cos(math.radians(latitude))
            zscale = 1 / meters_per_degree

            # Load the DEM and generate slope/aspect
            dem = rd.LoadGDAL(dem_path)
            slope = rd.TerrainAttribute(dem, 
                                        attrib='slope_percentage', 
                                        zscale=zscale).astype('float32')

            with rs.open(dem_path) as dem_r:
                profile = dem_r.profile
            profile.update(dtype='float32', count=1)

            with rs.open(slope_path, 'w', **profile) as dst:
                dst.write(slope, 1)
            slope_raster = rs.open(slope_path)

            if slope_raster.crs != prj.crs:
                print("CRS mismatch between raster and prj polygon")

            # get zonal stats for the entire prj at once
            aggregations = ['min', 'max', 'mean', 'median', 'majority']
            slope_stats_list = exact_extract(slope_raster, prj, aggregations)
            project_data = []
            for idx, (row, slope_zs) in enumerate(zip(prj.itertuples(), slope_stats_list)):
                
                if isinstance(slope_zs, dict) and 'properties' in slope_zs:
                    slope_stats = slope_zs['properties']
                else:
                    slope_stats = {stat: np.nan for stat in aggregations}
                # now calculate the area statistic for the decision
                area_stat = calculate_high_slope_area(slope_raster, 
                                                      row.geometry, 
                                                      threshold=slope_thresh)

                project_data.append({
                    'project_name': name,
                    'project_id': project_id,
                    'poly_id': getattr(row, 'poly_id'),
                    'slope_stats': slope_stats,
                    'slope_area': area_stat})
                
                df = pd.DataFrame(project_data).round(1)
                expanded = df["slope_stats"].apply(pd.Series).add_suffix("_slope")
                df = pd.concat([df.drop("slope_stats", axis=1), expanded], axis=1)
                df.to_csv(stat_path, index=False)
                dfs_to_concat.append(df)

            slope_raster.close()

        finally:
            # Delete all temp files
            for fpath in [dem_path, slope_path]:
                if os.path.exists(fpath):
                    os.remove(fpath)


    result_df = pd.concat(dfs_to_concat, ignore_index=True)

    # merge slope results into feats df (polys with missing slope become NaN)
    comb = feats_df.merge(result_df,
                          on=['project_id', 'poly_id'],
                          how='left')
    return comb
