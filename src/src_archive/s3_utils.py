import os
import subprocess
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import pandas as pd
import yaml
import boto3
import numpy as np

def list_s3_geojson_files(s3_geojson_path: str, project_name: str) -> list:
    """
    List all .geojson files for a project in the s3 geojson folder.
    """
    result = subprocess.run(['aws', 's3', 'ls', s3_geojson_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"[ERROR] Failed to list s3 path: {s3_geojson_path}\n{result.stderr}")
    
    geojson_files = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 4:
            filename = parts[3]
            if filename.endswith('.geojson') and filename.startswith(project_name):
                geojson_files.append(filename)
    
    return geojson_files

def get_latest_geojson_filename(files: list, project_name: str) -> str:
    """
    Returns the most recent geojson filename based on date in filename.
    """
    dated_files = []

    for filename in files:
        try:
            date_str = filename.replace(project_name + "_", "").replace(".geojson", "")
            file_date = datetime.strptime(date_str, "%m-%d-%Y")
            dated_files.append((file_date, filename))
        except ValueError:
            print(f"[WARN] Skipping file with invalid date format {filename}")
        
    if not dated_files:
        raise ValueError(f"No valid dated geojson files found for {project_name}")
    
    return max(dated_files)[1]

def download_geojson_from_s3(s3_path: str, local_path: Path):
    """
    Download a file from s3 to a local directory
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['aws', 's3', 'cp', s3_path, str(local_path)], check=True)

def load_geojson_as_gdf(local_path: Path, project_name: str) -> gpd.GeoDataFrame:
    """
    Load a local geojson as a GeoDataFrame with project info
    """
    gdf = gpd.read_file(local_path)
    gdf['project_name'] = project_name
    return gdf

def process_projects(projects: list, s3_base: str, local_base: str) -> gpd.GeoDataFrame:
    """
    For a list of projects (project shortnames), download the latest geojson and combine into one GeoDataFrame
    """
    combined_gdfs = []

    for project in projects:
        s3_geojson_path = f"{s3_base}{project}/geojson/"
        local_geojson_dir = Path(local_base) / project / "geojson"
        project_name = Path(project).name

        print(f"[INFO] Processing project {project}") 

        # Step 1: List and select latest file
        files = list_s3_geojson_files(s3_geojson_path, project)
        latest_file = get_latest_geojson_filename(files, project)

        # Step 2: Download
        s3_file_path = f"{s3_geojson_path}{latest_file}"
        local_file_path = local_geojson_dir / latest_file
        download_geojson_from_s3(s3_file_path, local_file_path)

        # Step 3: Load
        gdf = load_geojson_as_gdf(local_file_path, project)
        combined_gdfs.append(gdf)
    return gpd.GeoDataFrame(pd.concat(combined_gdfs, ignore_index=True))

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
    folder_path = f"{folder_prefix}/{project_name}/geojson/"
    s3_key = f"{folder_path}{filename}"
    local_path = os.path.join(params["outfile"]["geojsons"], filename)
    
    s3 = boto3.client('s3', 
                      aws_access_key_id=aak, 
                      aws_secret_access_key=ask)

    # Check if the project folder exists
    result = s3.list_objects_v2(Bucket=bucket, Prefix=folder_path, MaxKeys=1)
    if 'Contents' not in result:
        s3.put_object(Bucket=bucket, Key=folder_path)

    s3.upload_file(local_path, bucket, s3_key)
