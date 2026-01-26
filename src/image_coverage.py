import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import shape
from shapely.errors import ShapelyError
import ast

def clean_geometry(x, debug=True, stats=None):
    """
    Converts a geometry dict or string into a Shapely object. Attempts to fix invalid geometries using buffer(0). Returns None if geometry is still invalid, empty, or has 0 area.

    Args:
        x: Geometry dict or string
    
    Returns:
        Shapely geometry or None
    """
    # Initialize stats dictionary and increment the counter of total rows by 1
    if stats is not None:
        stats['total'] +=1

    # Try to convert a stringified dictionary or dictionary into a shapely object
    try:
        #geom = shape(ast.literal_eval(x)) if isinstance(x, str) else shape(x)
        geom = convert_geometry(x)
    except Exception as e:
        if debug:
            print(f"Failed to parse geometry: {e} -- {geom}")
        # Increment counter of dropped rows by 1
        if stats is not None:
            stats['invalid'] += 1
            stats['dropped'] +=1
        return None

    # Try to repair geometry if invalid
    if not geom.is_valid:
        # Increment counter of invalid geometries by 1
        if stats is not None:
            stats['invalid'] +=1
        # Try to repair by buffering by 0
        repaired = geom.buffer(0)
        # If buffer(0) fixed the geometry, return the repaired geometry
        if repaired.is_valid and not repaired.is_empty and repaired.area > 0:
            if debug:
                print(f"Geometry was repaired with buffer(0) -- {geom}")
            # Increment counter of repaired geometries by 1
            if stats is not None:
                stats['repaired'] += 1
            return repaired
        else:
            if debug:
                print(f"Geometry invalid and not repairable using buffer(0) -- {geom}")
            # Increment counter of dropped geometries by 1
            if stats is not None:
                stats['dropped'] +=1
            return None
        
    # Drop rows with empty geometries or geometries with 0 area
    if geom.is_empty or geom.area == 0:
        if debug:
            print(f"Geometry is empty or has zero area -- {geom}")
        # Increment counter of dropped geometries by 1
        if stats is not None:
            stats['dropped'] += 1
        return None
    
    return geom

def convert_geometry(poly_geom, debug=False):
    """
    Detects whether a geometry is in WKT, GeoJSON dictionary, or stringified GeoJSON-style dictionary format and converts it into a Shapely geometry object.  
    
    Args:
        poly_geom (str or dict) Geometry as a string (WKT or GeoJSON-style dictionary) or a dictionary.
    Returns:
        Shapely geometry object
    """
    if isinstance(poly_geom, str):
        poly_geom = poly_geom.strip() # Remove extra spaces

        if poly_geom.startswith('POLYGON') or poly_geom.startswith('MULTIPOLYGON'):
            # If it starts with POLYGON or MULTIPOLYGON, it's in WKT format
            if debug:
                print('Detected WKT format. Converting using wkt.loads()...')
            return wkt.loads(poly_geom)     
        else:
            try:
                # Assume stringified GeoJSON-style dictionary, convert string to dictionary first
                if debug:
                    print('Assuming GeoJSON format. Converting using ast.literal_eval() and shape()...')
                return shape(ast.literal_eval(poly_geom))
            except(SyntaxError, ValueError, TypeError, json.JSONDecodeError):
                print(f'Error: Could not parse geometry {poly_geom}')
                return None # Return None if conversion fails
    elif isinstance(poly_geom, dict):
        # If it's already a dictionary (GeoJSON format), use shape()
        return shape(poly_geom)
    
    return None # Return None if format is unrecognized

def preprocess_polygons(poly_df, debug=False, save_dropped=False, dropped_output_path=None):
    """
    Cleans up a dataframe of polygon metadata & geometries from the TerraMatch API and 
    converts it into a GeoDataframe. Tries to fix invalid geometries and drops (and logs) them if irreparable.

    Args:
        poly_df (DataFrame): Raw polygon dataset.
        save_dropped (bool): Save dropped polygons to a csv
        dropped_output_path (str): File path to save rows with dropped polygons

    Returns:
        GeoDataFrame: Processed polygon dataset with a geometry column as a shapely object.
    """
    print("Processing polygon data...")

    ## COLUMN NAMES
    # Enforce lowercase column names
    poly_df.columns = poly_df.columns.str.lower()
    # Rename 'name' and 'geometry' columns
    poly_df = poly_df.rename(columns={'name': 'poly_name', 'geometry': 'poly_geom'})
    
    ## PLANTSTART
    # Convert 'plantstart' column to a datetime
    poly_df['plantstart'] = pd.to_datetime(poly_df['plantstart'], errors='coerce')

    ## GEOMETRIES
    print("Cleaning geometries...")
    
    # Initialize stats to track # of invalid geometries
    stats = {
        'total': 0,
        'invalid': 0,
        'repaired': 0,
        'dropped': 0
    }

    # Check for invalid geometries, fix them if possible, and drop them if irreparable
    poly_df['poly_geom'] = poly_df['poly_geom'].apply(lambda x: clean_geometry(x, debug=debug, stats=stats))

    # Split dropped & clean polygons
    dropped_polys = poly_df[poly_df['poly_geom'].isna()]
    poly_df_clean = poly_df.dropna(subset=['poly_geom'])

    # Optionally save rows that were dropped due to invalid geometries
    if save_dropped and dropped_output_path and not dropped_polys.empty:
        dropped_polys.to_csv(dropped_output_path, index=False)
        print(f"Dropped polygons saved to {dropped_output_path}")

    # Create GeoDataFrame from cleaned data
    poly_gdf = gpd.GeoDataFrame(poly_df_clean, geometry='poly_geom', crs="EPSG:4326")

    # Summary
    print("\n🧾 Geometry Cleaning Summary:")
    print(f"  ➤ Total geometries processed: {stats['total']}")
    print(f"  ➤ Invalid geometries:         {stats['invalid']}")
    print(f"  ➤ Repaired with buffer(0):    {stats['repaired']}")
    print(f"  ➤ Dropped:                    {stats['dropped']}")
    print(f"  ✅ Final valid polygons:       {len(poly_gdf)}\n")

    return poly_gdf

def preprocess_images(img_df, debug=False):
    """
    Cleans up a dataframe of maxar image metadata & geometries from the Maxar Discovery API and 
    converts it into a GeoDataframe

    Args:
        img_df (DataFrame): Raw image metadata dataset.
    
    Returns: 
        GeoDataFrame: Processed image dataset with a geometry column as a shapely object.
    """
    print("Processing Maxar image data...")

    # Convert 'datetime' column to a datetime and rename
    img_df.loc[:, 'datetime'] = pd.to_datetime(img_df['datetime'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce') # Convert to datetime type
    img_df.loc[:, 'datetime'] = img_df['datetime'].apply(lambda x: x.replace(tzinfo=None) if pd.notna(x) else x)    # Remove time zone info
    
    # Rename the 'datetime' column to 'img_date'
    img_df = img_df.rename(columns={'datetime': 'img_date'}) # Rename the column img_date

    # Select the relevent columns from img_df
    img_df = img_df[['img_id', 'title', 'project_id', 'poly_id', 'img_date', 'area:cloud_cover_percentage', 'eo:cloud_cover', 'area:avg_off_nadir_angle', 'view:sun_elevation', 'img_geom']]

    # Convert stringified 'poly_geom' dictionaries into real dictionaries
    img_df['img_geom'] = img_df['img_geom'].apply(lambda x: shape(ast.literal_eval(x)) if isinstance(x, str) else shape(x))

    # Convert 'img_geom' (image footprint geometries) from WKT to Shapely objects
    img_df['img_geom'] = img_df['img_geom'].apply(shape)

    # Convert DataFrame to GeoDataFrame
    img_gdf = gpd.GeoDataFrame(img_df, geometry='img_geom', crs="EPSG:4326")

    print(f"There are {len(img_gdf)} images for {len(img_gdf.poly_id.unique())} polygons in {len(img_gdf.project_id.unique())} projects in this dataset.")

    return img_gdf

def merge_polygons_images(img_gdf, poly_gdf, debug=False):
    """ 
    Merges the polygon metadata into the Maxar image GeoDataFrame. All rows of the img_gdf are preserved.
    Also records polygons that are dropped because they don't have any associated images.

    Args:
        img_gdf (GeoDataFrame): Image metadata dataset (each row represents a Maxar image)
        poly_gdf (GeoDataFrame): Polygon dataset (each row represents a polygon from the TM API)
    
    Returns:
        tuple: (GeoDataFrame of merged dataset, list of missing polygons (poly_id, project_id))
    """
    print("Merging polygon metadata into image data...")

    # Merge the image data with the polygon data (preserving image data rows and adding associated polygon attributes)
    merged_gdf = img_gdf.merge(poly_gdf, on=['project_id', 'poly_id'], how='left')
    dropped_rows_invalid_poly = len(merged_gdf[merged_gdf['poly_geom'].isna()])

    # Filter out rows where polygon metadata is missing (i.e. no matching polygon found (polygon removed during preprocess_polygons because its geometry was invalid & unfixable))
    merged_gdf_clean = merged_gdf[merged_gdf['poly_geom'].notna()].copy()
    dropped_invalid_poly = len(merged_gdf['poly_id'].unique()) - len(merged_gdf_clean['poly_id'].unique())

    # Identify polygons without any corresponding Maxar images
    missing_polygons_df = poly_gdf[~poly_gdf['poly_id'].isin(merged_gdf_clean['poly_id'])]

    # Save poly_id and project_id of missing polygons as a list of tuples
    missing_polygons_list = list(missing_polygons_df[['poly_id', 'project_id']].itertuples(index=False, name=None))

    print(f"Total images in img_gdf: {len(img_gdf)}")
    print(f"Total polygons in poly_gdf: {len(poly_gdf)}")
    print(f"Number of polygons removed from merged dataset due to invalid (unfixable) geometries: {dropped_invalid_poly}")
    print(f"Number of rows removed from image dataset because their polygons had invalid (unfixable) geometries: {dropped_rows_invalid_poly}")
    print(f"Total rows in merged dataset: {len(merged_gdf_clean)}")
    print(f"Unique polygons in merged dataset: {len(merged_gdf_clean['poly_id'].unique())}")

    # Count polygons dropped due to no matching images
    missing_polygons = len(poly_gdf[~poly_gdf['poly_id'].isin(merged_gdf_clean['poly_id'])])
    print(f"{missing_polygons} polygons were dropped from the merged dataset because they have no Maxar images")
    print(f"Polygons without images (dropped at this stage): {missing_polygons_list}")

    return merged_gdf_clean, missing_polygons_list

def filter_images(merged_gdf, filters, debug=False):
    """
    Filters the merged dataset to retain only images that meet filters for image quality.
    The values for the filters can be changed in the parameters section.

    Args:
        merged_gdf (GeoDataFrame): Merged dataset of images and polygons.
        filters (dict): Dictionary containing filter thresholds (in Parameters section of notebook)
    
    Returns:
        GeoDataFrame: Filtered dataset containing only the images that meet the criteria
    """
    # Ensure date columns are in correct datetime format
    merged_gdf['img_date'] = pd.to_datetime(merged_gdf['img_date'], errors='coerce')
    merged_gdf['plantstart'] = pd.to_datetime(merged_gdf['plantstart'], errors='coerce')

    # Compute the date difference (image capture date - plant start date)
    merged_gdf['date_diff'] = (merged_gdf['img_date'] - merged_gdf['plantstart']).dt.days

    # Apply filtering criteria to retain only images within the desired time range, cloud cover, 
    # off nadir angle, and sun elevation parameters
    filtered_images = merged_gdf[
        (merged_gdf['date_diff'] >= filters['date_range'][0]) &
        (merged_gdf['date_diff'] <= filters['date_range'][1]) &
        (merged_gdf['area:cloud_cover_percentage'] < filters['cloud_cover']) &
        (merged_gdf['area:avg_off_nadir_angle'] <= filters['off_nadir']) &
        (merged_gdf['view:sun_elevation'] >= filters['sun_elevation'])
    ].copy()  # Copy to avoid SettingWithCopyWarning

    print(f"Total images before filtering: {len(merged_gdf)}")
    print(f"Total images after filtering: {len(filtered_images)}")
    print(f"Polygons with at least one valid filtered image: {filtered_images['poly_id'].nunique()}")

    return filtered_images

def filter_images_by_task(merged_gdf, global_filters, task_filters, debug=False):
    """
    Filters the merged dataset to retain only images that meet filters for image quality.
    Uses global filters for sun elevation angle & off-nadir angle and task-specific filters for baseline date range.
    The values for the global filters can be changed in the parameters section.

    Args:
        merged_gdf (GeoDataFrame): Merged dataset of images and polygons. MUST contain a task_id column (project_id_plantstart-year)
        global_filters (dict): Dictionary containing global filter thresholds (cloud cover, off nadir, sun elevationa angle, optional default date range) 
                               (from the Parameters section of notebook)
        task_filters (dict): Dictionary mapping the task_id to baseline date range filters for each task (imported in CSV)
                             task_id -> (date_range_start, date_range_end) (e.g. {'..._2021': (-366, 90), ...})
    
    Returns:
        GeoDataFrame: Filtered dataset containing only the images that meet the criteria
    """
    # Ensure date columns are in correct datetime format
    merged_gdf['img_date'] = pd.to_datetime(merged_gdf['img_date'], errors='coerce')
    merged_gdf['plantstart'] = pd.to_datetime(merged_gdf['plantstart'], errors='coerce')

    # Compute the date difference (image capture date - plant start date)
    merged_gdf['date_diff'] = (merged_gdf['img_date'] - merged_gdf['plantstart']).dt.days

    # Default date range if a task_id is missing from task_filters
    default_date_range = global_filters.get('date_range', (-366, 90)) # PPC default baseline

    # Map per-task date ranges onto each row
    # task_filters: task_id -> (start, end)
    date_ranges = merged_gdf['task_id'].map(lambda tid: task_filters.get(tid, default_date_range))

    merged_gdf['date_range_start'] = date_ranges.apply(lambda dr: dr[0])
    merged_gdf['date_range_end'] = date_ranges.apply(lambda dr: dr[1])

    # Apply filtering criteria to retain only images within the desired time range, cloud cover, 
    # off nadir angle, and sun elevation parameters
    #  - per-row date_range_start / date_range_end
    #  - global cloud_cover / off_nadir_angle / sun_elevation
    filtered_images = merged_gdf[
        (merged_gdf['date_diff'] >= merged_gdf['date_range_start']) &
        (merged_gdf['date_diff'] <= merged_gdf['date_range_end']) &
        (merged_gdf['area:cloud_cover_percentage'] < global_filters['cloud_cover']) &
        (merged_gdf['area:avg_off_nadir_angle'] <= global_filters['off_nadir']) &
        (merged_gdf['view:sun_elevation'] >= global_filters['sun_elevation'])
    ].copy()  # Copy to avoid SettingWithCopyWarning

    print(f"Total images before filtering: {len(merged_gdf)}")
    print(f"Total images after filtering: {len(filtered_images)}")
    print(f"Polygons with at least one valid filtered image: {filtered_images['poly_id'].nunique()}")

    return filtered_images

def get_best_image(poly_images, debug=False):
    """
    Selects the best image for a given polygon. Selects the most recent image with < 10% cloud cover. If no images have < 10% cloud cover, selects
    the image with the lowest cloud cover (using most recent image as a tiebreaker).

    If we want to update this to include an "expected coverage" based on cloud cover and footprint overlap,
    this is where it would go. Also potential to update to try the "next best" image if the "best" image covers < 50% of the polygon.

    Args:
        poly_images (GeoDataFrame): Subset of img_gdf_filtered containing images for one polygon.
    
    Returns:
        GeoSeries: The best image row from poly_images
    """
    # Filter poly_images to those with < 10% cloud cover (and sort by recency)
    low_cc_images = poly_images[poly_images['area:cloud_cover_percentage'] < 10].copy().sort_values(by=['img_date'], ascending=False)

    # If there is at least 1 image with < 10% cloud cover, return the most recent image as the best image
    if len(low_cc_images) > 0:
        if debug:
            print("There are images with < 10% cloud cover! Selecting most recent image.")
        best_image = low_cc_images.iloc[0]
        return best_image
    
    # If there are no images with < 10% cloud cover (but a nonzero amount with cloud cover under the hard threshold)
    else:
        if debug:
            print("There are no images with <10% cloud cover. Selecting image with lowest cloud cover.")
        # Sort images by cloud cover ascending (lower is better) and then by image date descending (more recent is better)
        sorted_images = poly_images.sort_values(by=['area:cloud_cover_percentage', 'img_date'], ascending=[True, False])
        
        # Return the most recent image as the best image
        best_image = sorted_images.iloc[0]
        return best_image

def get_utm_crs(longitude, latitude):
    """
    Determines the appropriate UTM CRS for a given polygon based on the longitude and latitude
    of its centroid.

    Args:
        longitude (float): Longitude of the polygon centroid.
        latitude (float): Latitude of the polygon centroid.
    
    Returns:
        str: EPSG code of the best UTM CRS.
    """

    # UTM zones range from 0 to 60, each covering 6 degrees of longitude
    utm_zone = int((longitude + 180) / 6) + 1

    # EPSG code for UTM Northern vs. Southern hemisphere
    epsg_code = 32600 + utm_zone if latitude >= 0 else 32700 + utm_zone

    return f"EPSG:{epsg_code}"

def compute_polygon_image_coverage(poly_id, project_id, poly_gdf, img_gdf_filtered,
                                   low_img_coverage_log, min_coverage_threshold=50, debug=False):
    """
    Computes the percentage of a polygon's area that is covered by its best available Maxar image.

    Args:
        poly_id (str): Unique polygon ID
        project_id (str): Unique project ID
        poly_gdf (GeoDataFrame): Polygons dataset
        img_gdf_filtered (GeoDataFrame): Images dataset filtered by hard criteria
        low_img_coverage_log (list): List to store cases where imagery coverage is low
        min_coverage_threshold (int): Imagery coverage threshold below which polygons are logged (defaults to 50%)

    Returns:
        Tuple containing (poly_id, project_id, best_image, img_date, num_images, poly_area_ha,
        overlap_area_ha, percent_img_cover)
    """

    ## Calculate the polygon's area from its geometry
    # Explicitly retrieve the polygon row
    polygon_row = poly_gdf.loc[poly_gdf['poly_id'] == poly_id]
    print(f"Computing coverage for polygon {poly_id}")

    # Retrieve the actual polygon geometry
    poly_geom = polygon_row['poly_geom'].values[0]
    if debug:
        print(f'poly_geom: {poly_geom}')

    # Compute UTM Zone based on polygon centroid and reproject polygon and image footprint
    poly_centroid = poly_geom.centroid
    if debug:
        print(f'poly_centroid: {poly_centroid}')
    utm_crs = get_utm_crs(poly_centroid.x, poly_centroid.y)
    if debug:
        print(f'Calculated UTM CRS: {utm_crs}')

    # Reproject the polygon geometry into a CRS with a unit of square meters
    poly_reprojected = gpd.GeoDataFrame(polygon_row, geometry='poly_geom', crs="EPSG:4326").to_crs(utm_crs)
    poly_geom_reprojected = poly_reprojected.geometry.iloc[0] # Extract reprojected polygon

    # Calculate polygon area dynamically (in hectares)
    poly_area_ha = poly_geom_reprojected.area / 10_000

    ## Get all images associated with this polygon
    poly_images = img_gdf_filtered[img_gdf_filtered['poly_id'] == poly_id]
    num_images = len(poly_images)

    ## Handle polygons with no valid images
    # If no valid images, log the case and return 0% imagery coverage
    if poly_images.empty:
        log_entry = {
            'poly_id': poly_id,
            'project_id': project_id,
            'best_image': None,
            'img_date': None,
            'num_images': 0,
            'poly_area_ha': poly_area_ha,
            'overlap_area_ha': 0,
            'percent_img_cover': 0,
        }
        print(f"Logging low covarage for polygon {poly_id} because there is no available imagery")
        low_img_coverage_log.append(log_entry)

        # Return after logging
        return (poly_id, project_id, None, None, num_images, poly_area_ha, 0, 0)
    
    ## If there is at least one valid image
    # Select the best image from the filtered images for the polygon
    best_image = get_best_image(poly_images)
    print(f"Found best image: {best_image}")

    # Retrieve the geometry of the best image footprint
    best_image_geom = best_image['img_geom']
    if debug:
        print(f"Best image geometry: {best_image_geom}")
    
    # Reproject the best image footprint's geometry
    best_img_reprojected = gpd.GeoDataFrame([best_image], geometry="img_geom", crs="EPSG:4326").to_crs(utm_crs)
    if debug:
        print(f"Reprojected best image to {utm_crs}")
    best_img_geom_reprojected = best_img_reprojected.geometry.iloc[0]

    # Compute intersection between polygon geometry and best image's footprint geometry
    overlap_area = poly_geom_reprojected.intersection(best_img_geom_reprojected).area
    overlap_area_ha = overlap_area / 10_000
    if debug:
        print(f"The best image overlaps {overlap_area_ha}ha of {poly_id}.")

    # Compute percentage of polygon covered by best image
    percent_img_cover = (overlap_area_ha / poly_area_ha) * 100
    print(f"Polygon {poly_id} has {percent_img_cover}% image cover")

    # Log cases where the imagery coverage of the best image is below the threshold
    if percent_img_cover < min_coverage_threshold or percent_img_cover == 0:
        print(f"Logging low coverage for polygon {poly_id}: {percent_img_cover}%")
        log_entry = {
            'poly_id': poly_id,
            'project_id': project_id,
            'best_image': best_image['title'] if best_image is not None else None,
            'img_date': best_image['img_date'] if best_image is not None else None,
            'num_images': num_images,
            'poly_area_ha': poly_area_ha,
            'overlap_area_ha': overlap_area_ha,
            'percent_img_cover': percent_img_cover,
        }
        low_img_coverage_log.append(log_entry)
    
    print("Success!")
    # Return results
    return (poly_id, project_id, best_image['title'], best_image['img_date'], num_images, poly_area_ha, overlap_area_ha,
            percent_img_cover)

def aggregate_project_image_coverage(results_df, debug=False):
    """
    Aggregates the polygon-level image coverage data to the project level.

    Args:
        results_df (DataFrame): Contains polygon-level image coverage data.
    
    Returns:
        project_coverage_df (DataFrame): Aggregated project-level imagery coverage summary
    """

    # Group data by project_id
    grouped = results_df.groupby('project_id')
    if debug:
        print(f"There are {len(grouped.project_id.unique())} projects being analyzed.")
    
    # Compute summary statistics per project
    project_coverage_df = grouped.agg(
        num_polygons=('poly_id', 'count'), # Total polygons
        num_polygons_with_images=('percent_img_cover', lambda x: (x > 0).sum()), # Polygons with imagery
        num_polygons_no_images=('percent_img_cover', lambda x: (x == 0).sum()), # Polygons with 0% imagery coverage
        total_project_area_ha=('poly_area_ha', 'sum'), # Total area of the project in hectares (sum of polygon areas)
        total_overlap_area_ha=('overlap_area_ha', 'sum'), # Total area of the project with imagery coverage (sum of per polygon area with imagery coverage)
    ).reset_index()

    # Compute the total percent area of each project that has Maxar imagery coverage (final result)
    project_coverage_df['total_percent_area_covered'] = (
        (project_coverage_df['total_overlap_area_ha'] / project_coverage_df['total_project_area_ha']) * 100
    )

    return project_coverage_df

def aggregate_project_image_coverage_ppc(results_df, debug=False):
    """
    Aggregates the polygon-level image coverage data to the task (PPC project_id + plantstart year of interest) level.

    Args:
        results_df (DataFrame): Contains polygon-level image coverage data. Should be filtered to only include polygons for years of interest.
    
    Returns:
        project_coverage_df (DataFrame): Aggregated project-level imagery coverage summary
    """
    # Group the data by project_id and plantstart_year
    grouped = results_df.groupby(['project_id', 'plantstart_year'])

    # Compute summary statistics per project
    project_coverage_df = grouped.agg(
        num_polygons=('poly_id', 'count'), # Total polygons
        num_polygons_with_images=('percent_img_cover', lambda x: (x > 0).sum()), # Polygons with imagery
        num_polygons_no_images=('percent_img_cover', lambda x: (x == 0).sum()), # Polygons with 0% imagery coverage
        total_project_area_ha=('poly_area_ha', 'sum'), # Total area of the project in hectares (sum of polygon areas)
        total_overlap_area_ha=('overlap_area_ha', 'sum'), # Total area of the project with imagery coverage (sum of per polygon area with imagery coverage)
    ).reset_index()

    # Compute the total percent area of each project that has Maxar imagery coverage (final result)
    project_coverage_df['total_percent_area_covered'] = (
        (project_coverage_df['total_overlap_area_ha'] / project_coverage_df['total_project_area_ha']) * 100
    )

    # Create a unique 'task_id' for each project_id and plantstart_year combo
    project_coverage_df['task_id'] = project_coverage_df['project_id'] + "_" + project_coverage_df['plantstart_year'].astype(str)

    # Reorder columns
    project_coverage_df = project_coverage_df[['project_id', 'task_id', 'plantstart_year', 'num_polygons', 'num_polygons_with_images', 'num_polygons_no_images', 
                                               'total_project_area_ha', 'total_overlap_area_ha', 'total_percent_area_covered']]

    return project_coverage_df