#!/usr/bin/python3
# conda activate r62GEDI

from osgeo import gdal
import numpy as np
import pandas as pd
import rasterio
import os
import sys
sys.path.append('../Functions')
from STANDARD_FUNCTIONS import runcmd

os.chdir('/mnt/locutus/remotesensing/r62/fire_danger/')

# Load raster files
target_raster = "./fire_danger_scores/reproj_20221231_20221231.tiff"
ndvi_raster = "./CA_FUELS_LC22_F40_220_REPROJECTED.tif"

with rasterio.open(target_raster) as src1, rasterio.open(ndvi_raster) as src2, rasterio.open(ndvi_raster) as src2:
	print("target_resolution:", (round(src1.res[0], 5), round(src1.res[1], 5)),
		  'NDVI_resolution:', (round(src2.res[0], 5), round(src2.res[1], 5)))

ndvi_tif = gdal.Open(ndvi_raster)
ndvi_resampled = gdal.Warp('./temporary/CA_FUELS_LC22_F40_220_REPROJECTED_RESAMPLED.tif', ndvi_tif, xRes = src1.res[0], yRes = src1.res[1])
