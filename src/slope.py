import os
import geopandas as gpd
import pandas as pd
import fiona
import pyproj
from osgeo import gdal
from PIL import Image
import matplotlib.pyplot as plt
import yaml



# https://github.com/wri/tree-verification/blob/main/notebooks/slope-calculator.ipynb

## takes in slope and aspect calculations per polygon, turns this into a decision
