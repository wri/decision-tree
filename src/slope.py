import os
import geopandas as gpd
import pandas as pd
import fiona
import pyproj
from osgeo import gdal
from PIL import Image
import georasters as gr
import matplotlib.pyplot as plt

import requests
from requests.auth import HTTPBasicAuth
import yaml

import elevation
import richdem as rd

import rasterio
from rasterio.plot import show
from rasterstats import zonal_stats

# https://github.com/wri/tree-verification/blob/main/notebooks/slope-calculator.ipynb

# TODO: maybe the workflow is split into 2 where the first part is
# captured in the data gathering phase (interaction with the API)
# and the actual calculation / thresholding is performed here.
# does it need to download the DEM data every time? Probably once.
# the branch code would actually take in the resulting csv file and
# determine what percent of polygon area with >20% slope. (threshold stored in params.yaml)

config_path = "../secrets.yaml"
with open(config_path) as conf_file:
    config = yaml.safe_load(conf_file)

project_list = [] # this is a list of project names

def shp_to_gdf(shapefile_path, driver):
    shapefile_features =[]
    with fiona.collection(shapefile_path, driver=driver) as shp_in:
        shp_crs = pyproj.crs.CRS(shp_in.crs)
        shpschema = shp_in.schema.copy()
        for obj in shp_in:
            shapefile_features.append(obj)
    shp_gdf = gpd.GeoDataFrame.from_features([feature for feature in shapefile_features], crs=shp_crs)
    shp_gdf['bounds'] = shp_gdf.geometry.apply(lambda x: x.bounds)
    shp_gdf['bounds'] = [list(i) for i in shp_gdf['bounds']]
    return shp_gdf

# gets the total bounds for every project
all_projects = {}
for project in project_list:
    shapefile= f"{os.path.join(data_path, project)}/{project}.shp"
    project_gdf = shp_to_gdf(shapefile, 'ESRI Shapefile')
    project_bounds = project_gdf.total_bounds
    query = {'demtype': 'NASADEM',
          'south': str(project_bounds[1]),
          'north': str(project_bounds[3]),
          'west': str(project_bounds[0]),
          'east': str(project_bounds[2]),
          'outputFormat': 'GTiff',
          'API_Key': str(config['key'])}
    with open('temp.tiff', 'wb') as f:
        ret = requests.get(config['dem_url'], stream=True, params=query)
        for data in ret.iter_content(1024):
            f.write(data)
    print(f"DEM downloaded for {project}")
    dem = rd.LoadGDAL('temp.tiff')
    slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
    aspect = rd.TerrainAttribute(dem, attrib='aspect')
    dem_r = rasterio.open('temp.tiff')
    affine = dem_r.transform

    project_dict = project_gdf.to_dict('records')

    for i in range(0, len(project_dict)):
        poly = project_dict[i]['geometry']
        project_dict[i]['slope_stats']= zonal_stats(poly, slope, affine=affine, stats=["min", "max", "mean", "median", "majority"])
        project_dict[i]['aspect_stats']= zonal_stats(poly, aspect, affine=affine, stats=["min", "max", "mean", "median", "majority"])
    all_projects[project] = project_dict


df_list = []
for key in all_projects.keys():
    project_df = pd.DataFrame.from_dict(all_projects[key], orient='columns')
    df_list.append(project_df)

all_projects_df = pd.concat(df_list)
(all_projects_df.loc[:, all_projects_df.columns != 'geometry']).to_csv("../data/polygon_slope_aspect.csv")
