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


def tm_pull_wrapper(url, headers, project_ids, outfile=None, geojson_dir=None):
    """
    Wrapper function around the TM API package.

    Iterates over a list of project IDs, aggregates results, and writes to JSON and GeoJSON if specified.

    Parameters:
        url (str): TerraMatch API endpoint.
        headers (dict): Auth headers.
        project_ids (list): List of project UUIDs.
        outfile (str): Optional path to write output JSON.
        geojson_dir (str): Optional directory to save project-level GeoJSONs.
    """
    all_results = []
    
    if geojson_dir:
        os.makedirs(geojson_dir, exist_ok=True)

    with tqdm(total=len(project_ids), desc="Pulling Projects", unit="project") as progress:
        for project_id in project_ids:
            params = {
                'projectId[]': project_id,
                'polygonStatus[]': 'approved',
                'includeTestProjects': 'false',
                'page[size]': '100'
            }

            try:
                #update this to directly use TM Api 
                results = patched_pull_tm_api_data(url, headers, params) 
                if results is None:
                    print(f"No results returned for project: {project_id}")
                    progress.update(1)
                    continue

                # Add project_id for traceability
                for r in results:
                    r['project_id'] = project_id

                all_results.extend(results)

                # --- Save GeoJSON per project if requested ---
                if geojson_dir:
             
                    gdf = gpd.GeoDataFrame(
                        results,
                        geometry=[shape(feature['geometry']) for feature in results],
                        crs='EPSG:4326'
                    )

                    # Validate geometries
                    # gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
                    # gdf = gdf[~gdf['geometry'].is_empty & gdf['geometry'].notnull()]
                    geojson_path = os.path.join(geojson_dir, f"{project_id}.geojson")
                    gdf.to_file(geojson_path, driver='GeoJSON')

            except Exception as e:
                print(f"Error pulling project {project_id}: {e}")

            progress.update(1)

    # Optional: write to JSON file
    if outfile:
        with open(outfile, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {outfile}")

    return all_results

def calculate_high_slope_area(slope_raster, polygon, threshold=20):

    '''
    Masks the slope raster with the polygon to determine the
    percentage of polygon area with steep slope.
    If there are no high slope pixels (>20%) then will see 0

    we need a certain number of inventory plots in each polygon to 
    reach our designated sampling rate, so if we could only put plots 
    on a small proportion of a project, it's unlikely we would reach 
    our desired sampling rate and field verification might not 
    accurately capture planting activities.
    '''
    # Mask the slope raster with the polygon
    out_image, out_transform = rasterio.mask.mask(slope_raster, [polygon], crop=True, nodata=np.nan)
    out_image = out_image[0] 

    # Flatten and remove NaNs
    slope_values = out_image.flatten()
    slope_values = slope_values[~np.isnan(slope_values)]

    if len(slope_values) == 0:
        return np.nan  # No data available for polygon #TODO: make this explicitly no data

    high_slope_pixels = np.sum(slope_values > threshold)
    total_pixels = len(slope_values)
    percentage = (high_slope_pixels / total_pixels) * 100
    return round(percentage, 1)

def opentopo_pull_wrapper(dem_url, api_key, input_df, geojson_dir, outfile):
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
             tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as slope_tmp, \
             tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as aspect_tmp:
            
            dem_path = dem_tmp.name
            slope_path = slope_tmp.name
            aspect_path = aspect_tmp.name

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
            aspect = rd.TerrainAttribute(dem, attrib='aspect').astype('float32')

            # Load DEM to get spatial profile
            with rs.open(dem_path) as dem_r:
                profile = dem_r.profile
            profile.update(dtype='float32', count=1)

            with rs.open(slope_path, 'w', **profile) as dst:
                dst.write(slope, 1)
            with rs.open(aspect_path, 'w', **profile) as dst:
                dst.write(aspect, 1)

            # Open the rasters for reading
            slope_raster = rs.open(slope_path)
            aspect_raster = rs.open(aspect_path)

            aggregations = ['min', 'max', 'mean', 'median', 'majority']

            # Get zonal stats for all polygons at once
            slope_stats_list = exact_extract(slope_raster, prj, aggregations)
            aspect_stats_list = exact_extract(aspect_raster, prj, aggregations)

            for idx, (row, slope_zs, aspect_zs) in enumerate(zip(prj.itertuples(), slope_stats_list, aspect_stats_list)):
                if isinstance(slope_zs, dict) and 'properties' in slope_zs:
                    slope_stats = slope_zs['properties']
                else:
                    slope_stats = {stat: np.nan for stat in aggregations}

                if isinstance(aspect_zs, dict) and 'properties' in aspect_zs:
                    aspect_stats = aspect_zs['properties']
                else:
                    aspect_stats = {stat: np.nan for stat in aggregations}
                
                area_stat = calculate_high_slope_area(slope_raster, row.geometry, threshold=20) # move threshold to PARAMS

                all_prjs.append({
                    'project_id': project_id,
                    'poly_id': getattr(row, 'poly_id'),
                    'slope_stats': slope_stats,
                    'aspect_stats': aspect_stats,
                    'slope_area': area_stat
                })

            slope_raster.close()
            aspect_raster.close()

        finally:
            # Delete all temp files
            for fpath in [dem_path, slope_path, aspect_path]:
                if os.path.exists(fpath):
                    os.remove(fpath)

    # Write results to CSV
    result_df = pd.DataFrame(all_prjs)
    result_df = result_df.round(1)

    # Expand the nested dictionaries
    result_df = pd.concat([
        result_df.drop(['slope_stats', 'aspect_stats'], axis=1),
        result_df['slope_stats'].apply(pd.Series).add_suffix('_slope'),
        result_df['aspect_stats'].apply(pd.Series).add_suffix('_aspect')
    ], axis=1)

    result_df.to_csv(outfile, index=False)
