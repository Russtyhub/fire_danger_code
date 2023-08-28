#!/usr/bin/python3
# conda activate r62GEDI

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from osgeo import gdal
import numpy as np
import pandas as pd
import rasterio
import os
import sys
from datetime import date, timedelta
from itertools import chain
import time as TM

sys.path.append('../Functions')
from STANDARD_FUNCTIONS import runcmd, create_list_of_dates, format_with_zeros

os.chdir('/mnt/locutus/remotesensing/r62/fire_danger/')

fire_scores_dir = '/mnt/locutus/remotesensing/r62/fire_danger/fire_danger_scores/'
MODIS_dir = '/mnt/locutus/remotesensing/r62/fire_danger/MODIS_NDVI_IMPORT/'
daymet_dir = '/mnt/locutus/remotesensing/r62/fire_danger/daymet/'
temp_tif_dir = '/mnt/locutus/remotesensing/r62/fire_danger/temporary/'

# FUNCTIONS

def find_bbox_within_tif(tif, band=1):
    bbox = [] # top, bottom, left, right
    data = tif.read(band)
    data[data == -9999.0] = np.nan

    for idx, row in enumerate(data):
        if np.isnan(np.nanmin(row)):
            pass
        else:
            bbox.append(idx)
            break

    for idx, row in enumerate(data[::-1]):
        if np.isnan(np.nanmin(row)):
            pass
        else:
            if idx == 0:
                bbox.append(1)
                break
            else:
                bbox.append(idx)
                break 

    for idx, row in enumerate(data[::-1].T):
        if np.isnan(np.nanmin(row)):
            pass
        else:
            bbox.append(idx)
            break

    for idx, row in enumerate(data.T[::-1]):
        if np.isnan(np.nanmin(row)):
            pass
        else:
            if idx == 0:
                bbox.append(1)
                break
            else:
                bbox.append(idx)
                break

    data = data[bbox[0]:-bbox[1], bbox[2]:-bbox[3]]
    data[np.isnan(data)] = -9999.0
    # print(bbox)
    return data

def find_closest_number(arr, target):
    arr = np.array(arr)
    closest_idx = np.abs(arr - target).argmin()
    # closest_number = arr[closest_idx]
    return closest_idx

# Sample Random Fire Danger Scores for Resolution
target_raster = f"{fire_scores_dir}reproj_20221231_20221231.tiff"
with rasterio.open(target_raster) as src1:
    print("target_resolution:", (round(src1.res[0], 5), round(src1.res[1], 5)))

fuel_values_tif = rasterio.open('/mnt/locutus/remotesensing/r62/fire_danger/temporary/CA_FUELS_LC22_F40_220_REPROJECTED_RESAMPLED.tif')
fuel_values = fuel_values_tif.read(1)
# print('FUEL SCORES:', fuel_values.shape)

# Creating Time Sequence

start_date = date(2021, 6, 15)
end_date = date(2022, 12, 31)
days_list = create_list_of_dates(start_date, end_date, x_days=1)

start = TM.time()
for DAY in days_list:
    
    # Create a grid that is used for labeling. Then add that to our data list
    labels_grid = np.arange(fuel_values.shape[0]*fuel_values.shape[1]).reshape(fuel_values.shape)
    data = [labels_grid, fuel_values]
    # Get temporal statistics
    day_of_year = format_with_zeros(DAY.timetuple().tm_yday, 3)
    year = str(DAY.year)
    month = format_with_zeros(DAY.month, 2)
    day_of_month = format_with_zeros(DAY.day, 2)
    # Adding the radians 
    rads = (int(day_of_year)-1)*(360/364)*(np.pi / 180)
    data.append(np.cos(rads)*np.ones(fuel_values.shape))
    data.append(np.sin(rads)*np.ones(fuel_values.shape))

    # FIRE DANGER SCORES

    if os.path.exists(f'{fire_scores_dir}reproj_{year}{month}{day_of_month}_{year}{month}{day_of_month}.tiff'):
        fire_danger_score_tif = rasterio.open(f'{fire_scores_dir}reproj_{year}{month}{day_of_month}_{year}{month}{day_of_month}.tiff')
        fire_danger_score_data = find_bbox_within_tif(fire_danger_score_tif)
        data.append(fire_danger_score_data)
        # print('FIRE DANGER DATA SHAPE:', fire_danger_score_data.shape)
    else:
        continue


    # NDVI 

    ndvi_year = [i for i in os.listdir(MODIS_dir) if i.startswith(f'MOD13Q1.061__250m_16_days_NDVI_doy{year}')]
    doy_by_year=[]
    for i in ndvi_year:
        i = i.replace(f'MOD13Q1.061__250m_16_days_NDVI_doy{year}', '')
        i = int(i.split('_')[0])
        doy_by_year.append(i)

    closest_doy_idx = find_closest_number(doy_by_year, int(day_of_year))
    closest_tif_name = ndvi_year[closest_doy_idx]

    if os.path.exists(f'{temp_tif_dir}{closest_tif_name}'):
        closest_ndvi_tif = rasterio.open(f'{temp_tif_dir}{closest_tif_name}')
        closest_ndvi_data = np.squeeze(closest_ndvi_tif.read())
        data.append(closest_ndvi_data)
        # print('NDVI SHAPE', closest_ndvi_data.shape)

    else:
        closest_ndvi_tif = gdal.Open(f'{MODIS_dir}{closest_tif_name}')
        gdal.Warp(f'{temp_tif_dir}{closest_tif_name}', closest_ndvi_tif, xRes = src1.res[0], yRes = src1.res[1])
        closest_ndvi_tif = rasterio.open(f'{temp_tif_dir}{closest_tif_name}')
        closest_ndvi_data = np.squeeze(closest_ndvi_tif.read())
        data.append(closest_ndvi_data)
        # print('NDVI SHAPE', closest_ndvi_data.shape)

    # DAYMET 

    daymet_files = [i for i in os.listdir(daymet_dir) if i.endswith('.tif') and (i.split('_')[1] == year)]
    daymet_files.sort()
    for f in daymet_files:
        if os.path.exists(f'{temp_tif_dir}{f}'):
            FILE = rasterio.open(f'{temp_tif_dir}{f}')
            var_data = FILE.read(int(day_of_year))
            data.append(var_data)
        else:
            FILE = gdal.Open(f'{daymet_dir}{f}')
            gdal.Warp(f'{temp_tif_dir}{f}', FILE, xRes = src1.res[0]-0.00002, yRes = src1.res[1]-0.00002) # -0.00002
            FILE = rasterio.open(f'{temp_tif_dir}{f}')
            var_data = FILE.read(int(day_of_year))
            data.append(var_data)

    data = np.stack(data, axis=0)
    # print(data.shape)
    # Columns order: Index label, cos_rads, sin_rads, fire_danger_score_data, NDVI, prcp, srad, swe, tmax, tmin, vp

    np.save(f'/mnt/locutus/remotesensing/r62/fire_danger/binary_data_maps/{month}-{day_of_month}-{year}.npy', data[1:, ...])
    print(f'FILE: {month}-{day_of_month}-{year}.npy SAVED')
end = TM.time()
complete = (end-start)/60
print('COMPLETE TIME ELAPSED:', round(complete, 2), 'MINUTES')