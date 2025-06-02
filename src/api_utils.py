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
from rasterstats import zonal_stats
import richdem as rd
import tempfile
import ast
from shapely.geometry import shape
from osgeo import gdal

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


def TM_pull_wrapper(url, headers, project_ids, outfile=None):
    """
    Wrapper function around the TM API package.

    Iterates over a list of project IDs, aggregates results, and writes to JSON if specified.

    Parameters:
        url (str): TerraMatch API endpoint.
        headers (dict): Auth headers.
        project_ids (list): List of project UUIDs.
        outfile (str): Optional path to write output JSON.
    """
    all_results = []

    with tqdm(total=len(project_ids), desc="Pulling Projects", unit="project") as progress:
        for project_id in project_ids:
            params = {
                'projectId[]': project_id,
                'polygonStatus[]': 'approved',
                'includeTestProjects': 'false',
                'page[size]': '100'
            }

            try:
                results = patched_pull_tm_api_data(url, headers, params)
                # results = pull_tm_api_data(url, headers, params) confirm this now works

                if results is None:
                    print(f"No results returned for project: {project_id}")
                    continue

                for r in results:
                    r['project_id'] = project_id  # Add project_id for traceability

                all_results.extend(results)

            except Exception as e:
                print(f"Error pulling project {project_id}: {e}")

            progress.update(1)

    # Optional: write to JSON file
    if outfile:
        with open(outfile, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {outfile}")

    return all_results


def opentopo_pull_wrapper(dem_url, api_key, df, outfile):
    '''
    Downloads DEM data using the project bounding box, then calculates 
    slope and aspect statistics at the polygon level to ensure a yes/no
    decision can be made for each polygon.

    If a polygon is too small or falls outside DEM extent, stats are set to NaN.
    '''
    #df = pd.read_csv(project_csv) update this later to read in clean csv
    df['geometry'] = df['geometry'].apply(ast.literal_eval).apply(shape)

    projects = df['project_id'].unique()

    all_polygons = []

    for id in tqdm(projects, desc='Processing Projects', unit='project'):
        project_df = df[df.project_id == id].copy()
        gdf = gpd.GeoDataFrame(project_df, geometry='geometry', crs='EPSG:4326')

        # Project-level bounding box
        total_bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

        query = {
            'demtype': 'NASADEM',
            'south': str(total_bounds[1]),
            'north': str(total_bounds[3]),
            'west': str(total_bounds[0]),
            'east': str(total_bounds[2]),
            'outputFormat': 'GTiff',
            'API_Key': api_key
        }
        # Use a temporary file instead of manual file path
        # keep delete = False because file is used later
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
            temp_path = tmpfile.name

        try:
            # Download the DEM to the temp file
            response = requests.get(dem_url, stream=True, params=query)
            if response.status_code != 200:
                print(f"Error downloading DEM for project {id}: {response.text}")
                raise Exception(f"Failed to download DEM for project {id}")

            # Save the DEM content to the temp file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            # Load the DEM
            dem = rd.LoadGDAL(temp_path)
            slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
            aspect = rd.TerrainAttribute(dem, attrib='aspect')

            # Open the DEM with rasterio to get affine
            with rs.open(temp_path) as dem_r:
                affine = dem_r.transform

            for idx, row in gdf.iterrows():
                poly = row['geometry']

                # Initialize stats as NaN
                slope_stats = {stat: np.nan for stat in ["min", "max", "mean", 
                                                         "median", "majority"]}
                aspect_stats = {stat: np.nan for stat in ["min", "max", "mean", 
                                                          "median", "majority"]}

                try:
                    slope_zs = zonal_stats(poly, slope, affine=affine,
                                           stats=["min", "max", "mean", 
                                                  "median", "majority"])
                    aspect_zs = zonal_stats(poly, aspect, affine=affine,
                                            stats=["min", "max", "mean", 
                                                   "median", "majority"])
                    # Update stats if available
                    if slope_zs and slope_zs[0] is not None:
                        slope_stats = slope_zs[0]
                    if aspect_zs and aspect_zs[0] is not None:
                        aspect_stats = aspect_zs[0]
                except Exception as e:
                    print(f"Warning: Unable to calculate stats for polygon {row['poly_id']} in project {id}: {e}")

                all_polygons.append({
                    'project_id': id,
                    'poly_id': row['poly_id'],
                    'slope_stats': slope_stats,
                    'aspect_stats': aspect_stats
                })
        # now delete the temp file
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Write results to CSV
    result_df = pd.DataFrame(all_polygons)
    result_df = pd.concat([result_df.drop(['slope_stats', 'aspect_stats'], axis=1),
                       result_df['slope_stats'].apply(pd.Series).add_suffix('_slope'),
                       result_df['aspect_stats'].apply(pd.Series).add_suffix('_aspect')], axis=1)
    result_df.to_csv(outfile, index=False)
