import requests
import yaml
import pandas as pd
import os
import json
import numpy as np
import sys
import utm
import geopandas as gpd
import rasterio as rs
import rasterio.mask
import tempfile
import math

from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from contextlib import contextmanager
from typing import Union
from pyproj import CRS
from shapely.geometry import box, shape
from tqdm import tqdm
from exactextract import exact_extract

from tm_api_utils import pull_tm_api_data
from decision_tree_src.s3_utils import upload_to_s3
from decision_tree_src.tools import get_gfw_access_token, get_opentopo_api_key


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
    access_token = auth['gfw_api']['access_token']
    headers = {'Authorization': f"Bearer {access_token}"}
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

def opentopo_pull_wrapper(params, config, feats_df, process_in_utm_coordinates: bool = True):
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
    api_key = get_opentopo_api_key(config)
    geojson_dir = params['outfile']['geojsons']
    slope_thresh = params['criteria']['slope_thresh']
    data_version = params['outfile']['data_version']

    project_names = feats_df['project_name'].unique()
    project_names = [i for i in project_names if i != 'MLI_22_ASIC'] # still has an erroneous polygon that breaks the pipeline
    dfs_to_concat = []
    for name in project_names:
        this_project = []
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
        project_polygons = gpd.read_file(geojson_path)
        total_bounds = project_polygons.total_bounds
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
                for chunk in response.iter_content(8192): # 8k generally works well for chunking of data.
                    f.write(chunk)

            if process_in_utm_coordinates:
                site_polygons, slope_raster = _prepare_utm_features(project_polygons, dem_path)
            else:
                site_polygons, slope_raster = _prepare_latlon_features(total_bounds, project_polygons, dem_path)

            if slope_raster.crs != site_polygons.crs:
                print("CRS mismatch between raster and prj polygon")

            # get zonal stats for the entire prj at once
            aggregations = ['min', 'max', 'mean', 'median', 'majority']
            slope_stats_list = exact_extract(slope_raster, site_polygons, aggregations)

            for idx, (row, slope_zs) in enumerate(zip(site_polygons.itertuples(), slope_stats_list)):
                
                if isinstance(slope_zs, dict) and 'properties' in slope_zs:
                    slope_stats = slope_zs['properties']
                else:
                    slope_stats = {stat: np.nan for stat in aggregations}
                # now calculate the area statistic for the decision
                area_stat = calculate_high_slope_area(slope_raster, 
                                                      row.geometry, 
                                                      threshold=slope_thresh)

                project_data = []
                project_data.append({
                    'project_name': name,
                    'project_id': project_id,
                    'poly_id': getattr(row, 'poly_id'),
                    'slope_stats': slope_stats,
                    'slope_area': area_stat})
                
                df = pd.DataFrame(project_data).round(1)
                expanded = df["slope_stats"].apply(pd.Series).add_suffix("_slope")
                df = pd.concat([df.drop("slope_stats", axis=1), expanded], axis=1)

                # Append project results to composite list of df
                dfs_to_concat.append(df.copy())
                this_project.append(df.copy())

            # Cache project results to file
            project_results_df = pd.concat(this_project, ignore_index=True)
            project_results_df.to_csv(stat_path, index=False)

            slope_raster.close()

        finally:
            # Delete all temp files
            for fpath in [dem_path, slope_path]:
                if os.path.exists(fpath):
                    os.remove(fpath)

    # Convert list of dataframes to result dataframe
    result_df = pd.concat(dfs_to_concat, ignore_index=True)

    # merge slope results into feats df (polys with missing slope become NaN)
    comb = feats_df.merge(result_df,
                          on=['project_id', 'poly_id'],
                          how='left')

    return comb


def _prepare_latlon_features(total_bounds, project_polygons, dem_path):
    # convert degrees to meters
    latitude = (total_bounds[1] + total_bounds[3]) / 2
    meters_per_degree = 111320 * math.cos(math.radians(latitude))
    z_factor = 1 / meters_per_degree

    # Read DEM into memory
    try:
        dem_memory_file = _load_dem_to_memoryfile(dem_path)

        slope_raster = _compute_slope_percent_from_memory(dem_memory_file, z_factor, Resampling.bilinear)

    except Exception as ex:
        print(ex)

    finally:
        # close memory files
        dem_memory_file.close()

    return project_polygons, slope_raster


def _prepare_utm_features(project_polygons, dem_path):
    z_factor = 1

    # determine best utm projection
    with rs.open(dem_path) as dem_r:
        src_bbox = [dem_r.bounds[0], dem_r.bounds[1], dem_r.bounds[2], dem_r.bounds[3]]
    dst_crs = _get_utm_zone_epsg(src_bbox)

    try:
        # Read DEM into memory and project to uTM
        dem_memory_file = _load_dem_to_memoryfile(dem_path)
        reprojected_mem_file = _reproject_raster_in_memory(dem_memory_file, dst_crs)

        slope_raster = _compute_slope_percent_from_memory(reprojected_mem_file, z_factor, Resampling.bilinear)

    except Exception as ex:
        print(ex)

    finally:
        # close memory files
        dem_memory_file.close()
        reprojected_mem_file.close()

    projected_project_polygons = project_polygons.to_crs(dst_crs)

    return projected_project_polygons, slope_raster


def _load_dem_to_memoryfile(dem_path: str):
    """
    Reads a DEM file from disk into a rasterio MemoryFile and returns the MemoryFile.
    The caller is responsible for closing the returned MemoryFile.
    """
    # Read DEM bytes from disk
    with open(dem_path, "rb") as dem_r:
        dem_bytes = dem_r.read()

    # Create MemoryFile containing the DEM
    memfile = MemoryFile(dem_bytes)

    return memfile


def _compute_slope_percent_from_memory(
    dem_source: Union[MemoryFile, rasterio.io.DatasetReader],
    z_factor: float = 1.0,
    resampling: Resampling = Resampling.bilinear
) -> rasterio.io.DatasetReader:
    """
    Compute slope (%) from an in-memory DEM using Horn's method.

    Parameters
    ----------
    dem_source : rasterio.io.MemoryFile | rasterio.io.DatasetReader
        In-memory DEM raster. Can be a MemoryFile or an already-open DatasetReader.
        Must be a single-band elevation raster (or first band is elevation).
    z_factor : float
        Ratio of vertical units to horizontal units.
        Example: Elevation in feet, horizontal in meters -> z_factor = 0.3048
    resampling : rasterio.enums.Resampling
        Resampling method for reading DEM.

    Returns
    -------
    rasterio.io.DatasetReader
        An open dataset reader for a new in-memory raster with slope in percent.
        The caller is responsible for closing it when done.

    Notes
    -----
    - Slope (%) = sqrt((dz/dx)^2 + (dz/dy)^2) * 100
    - Uses Horn 3x3 operator for dz/dx and dz/dy.
    """

    @contextmanager
    def _ensure_reader(src_obj):
        """
        Context manager that yields a DatasetReader from either a MemoryFile or
        an existing DatasetReader. If src_obj is a MemoryFile, opens it here.
        If it's already a DatasetReader, yields it as-is without closing it.
        """
        needs_close = False
        if isinstance(src_obj, MemoryFile):
            src = src_obj.open()
            needs_close = True
        elif isinstance(src_obj, rasterio.io.DatasetReader):
            src = src_obj
        else:
            raise TypeError(
                "dem_source must be a rasterio MemoryFile or DatasetReader."
            )
        try:
            yield src
        finally:
            if needs_close:
                src.close()

    with _ensure_reader(dem_source) as src:
        # Read DEM band 1 with requested resampling
        dem = src.read(1, resampling=resampling).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
        profile = src.profile

        # Pixel sizes (absolute values)
        cellsize_x = transform.a
        cellsize_y = -transform.e  # transform.e is typically negative for north-up rasters

        if not np.isfinite(cellsize_x) or not np.isfinite(cellsize_y) or cellsize_x <= 0 or cellsize_y <= 0:
            raise ValueError("Invalid pixel size from transform.")

        # Build mask (true where NoData)
        if nodata is not None and np.isfinite(nodata):
            mask = (dem == nodata) | ~np.isfinite(dem)
        else:
            mask = ~np.isfinite(dem)

        # Apply z-factor
        dem_scaled = dem * z_factor

        # Pad for 3x3 neighborhood using edge replication
        padded = np.pad(dem_scaled, pad_width=1, mode='edge')

        # Neighbors (Horn)
        z1 = padded[:-2, :-2]; z2 = padded[:-2, 1:-1]; z3 = padded[:-2, 2:]
        z4 = padded[1:-1, :-2];                           z6 = padded[1:-1, 2:]
        z7 = padded[2:,  :-2]; z8 = padded[2:,  1:-1];  z9 = padded[2:,  2:]

        # Partial derivatives
        dzdx = ((z7 + 2.0*z8 + z9) - (z1 + 2.0*z2 + z3)) / (8.0 * cellsize_x)
        dzdy = ((z3 + 2.0*z6 + z9) - (z1 + 2.0*z4 + z7)) / (8.0 * cellsize_y)

        # Gradient magnitude -> percent
        slope_percent = np.sqrt(dzdx**2 + dzdy**2) * 100.0

        # Apply mask
        # Keep output nodata consistent: if src had nodata, use it; otherwise use NaN
        out_nodata = nodata if (nodata is not None) else np.nan
        slope_percent[mask] = out_nodata

        # Update raster profile for the output
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=out_nodata
        )

    # ---- Write and return as a new in‑memory raster ----
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(slope_percent.astype(np.float32), 1)

    # Return an open dataset reader (caller should close it when done)
    return memfile.open()


def _get_utm_zone_epsg(bbox):
    """
    Get the UTM zone projection for given a bounding box.

    :param bbox: tuple of (min x, min y, max x, max y)
    :return: the EPSG code for the UTM zone of the centroid of the bbox
    """
    centroid = box(*bbox).centroid
    utm_x, utm_y, band, zone = utm.from_latlon(centroid.y, centroid.x)

    if centroid.y > 0:  # Northern zone
        epsg = 32600 + band
    else:
        epsg = 32700 + band

    return CRS.from_string(f"EPSG:{epsg}")


def _reproject_raster_in_memory(input_memfile, dst_crs):
    """
    Reproject a raster stored in a rasterio MemoryFile into another projection.

    Parameters:
        input_memfile (MemoryFile): MemoryFile containing the source raster.
        dst_crs (str or dict): Target CRS (e.g., 'EPSG:3857').

    Returns:
        MemoryFile: A new MemoryFile containing the reprojected raster.
    """

    # Open source raster from memory
    with input_memfile.open() as src:
        # Compute new transform and shape
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        # Create output MemoryFile
        output_memfile = MemoryFile()
        with output_memfile.open(**kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

    # IMPORTANT: return the MemoryFile itself — not the open dataset
    return output_memfile
