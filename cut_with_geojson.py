#!/usr/bin/python3

import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
from osgeo import gdal
from rasterio.mask import mask
import geopandas as gpd
import sys
import os
sys.path.append('/home/r62/repos/russ_repos/Functions')
os.chdir('/mnt/locutus/remotesensing/r62/fire_danger')

from REMOTE_SENSING_FUNCTIONS import clip_tiff_with_geojson, clip_raster_using_smaller_raster

clip_tiff_with_geojson('./reproj_20210620_20210620.tiff', './California_State_Boundary/California_State_Boundary.geojson', './reproj_20210620_20210620_clipped.tiff')
clip_raster_using_smaller_raster('./MODIS_NDVI_IMPORT/MOD13Q1.061__250m_16_days_NDVI_doy2019353_aid0001.tif', './reproj_20210620_20210620_clipped.tiff', './reproj_20210620_20210620_clipped.tiff')