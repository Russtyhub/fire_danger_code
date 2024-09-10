#!/usr/bin/python3
# conda activate r62GEDI

import rasterio 
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import geopandas as gpd
import os
import numpy as np
from datetime import datetime, date
import sys
import time as TM

sys.path.append('/home/r62/repos/russ_repos/Functions')
from TIME import create_list_of_dates
from STANDARD_FUNCTIONS import runcmd, do_all_files_exist
PRINT = False

# Columns order: cos_rads, sin_rads, fire_danger_score, NDVI, prcp, srad, swe, tmax, tmin, vp
daymet_vars = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

def clip_tiff_with_geojson(tiff_path, geojson_path, output_path):
    # Read the GeoJSON file using geopandas
    gdf = gpd.read_file(geojson_path)

    # Make sure the GeoJSON is in the same CRS as the TIFF
    with rasterio.open(tiff_path) as src:
        gdf = gdf.to_crs(src.crs)

    # Combine all geometries into a single MultiPolygon
    multi_poly = gdf.unary_union

    # Open the TIFF file
    with rasterio.open(tiff_path) as src:
        # Crop the TIFF to the MultiPolygon
        out_image, out_transform = mask(src, shapes=[multi_poly], crop=True)

        # Copy the metadata from the original TIFF
        out_meta = src.meta.copy()

        # Update the metadata with new dimensions and transform
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # Write the clipped TIFF to the output file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def update_tiff(input_tif, reproj_tiff):
    with rasterio.open(input_tif) as src:

        metadata = src.meta
        data = src.read() 
        
        metadata.update({
            'transform': new_transform,
            'width': fire_danger_meta['width'],
            'height': fire_danger_meta['height'],
            'crs' : fire_danger_meta['crs'],
            'dtype' : 'float32',
            'nodata' : -9999.0,
        })

        with rasterio.open(reproj_tiff, 'w', **metadata) as dst:
            for idx, i in enumerate(data):
                idx = idx + 1
                dst.write(i, idx) 

def find_closest_number(arr, target):
    arr = np.array(arr)
    closest_idx = np.abs(arr - target).argmin()
    # closest_number = arr[closest_idx]
    return closest_idx

# Defining paths:
geojson_path = '/mnt/locutus/remotesensing/r62/fire_danger/California_State_Boundary/California_State_Boundary.geojson'
fire_danger_path = '/mnt/locutus/remotesensing/r62/fire_danger/fire_danger_scores_V2'
MODIS_path = '/mnt/locutus/remotesensing/r62/fire_danger/MODIS_NDVI_IMPORT_V2'
daymet_path = '/mnt/locutus/remotesensing/r62/fire_danger/daymet'
temporary_path = '/mnt/locutus/remotesensing/r62/fire_danger/temporary'
binary_maps_path = '/mnt/locutus/remotesensing/r62/fire_danger/binary_data_maps_V2'

start_date = date(2020, 1, 1)
end_date = date(2023, 12, 31)
days_list = create_list_of_dates(start_date, end_date, x_days=1)

start = TM.time()

# I want to map my values to the same resolution as the fire potential index
# Below I take an example tiff and record the new transform I will need to apply
# to the upcoming tiffs
print('EXAMPLE FIRE DANGER POTENTIAL DATA')

input_fire_danger_tif = '/mnt/locutus/remotesensing/r62/fire_danger/fire_danger_scores_V2/reproj_20211119_20211119.tiff'
output_fire_danger_tif = '/mnt/locutus/remotesensing/r62/fire_danger/fire_danger_scores_V2/reproj_edges_clipped.tiff'

clip_tiff_with_geojson(input_fire_danger_tif, geojson_path, output_fire_danger_tif)

fire_danger_tif = rasterio.open(output_fire_danger_tif)
fire_danger_meta = fire_danger_tif.meta
if PRINT:
    print(fire_danger_meta)
    print()
# Define the new transform
new_transform = from_origin(fire_danger_tif.transform[2], 
                            fire_danger_tif.transform[5], 
                            fire_danger_tif.transform[0], 
                            fire_danger_tif.transform[0])

runcmd(f'rm {output_fire_danger_tif}')

# Using transform and meta data on the fuels tiff
fuels_tif = '/mnt/locutus/remotesensing/r62/fire_danger/fuels/fuels_resampled.tif'
update_tiff(fuels_tif, f'{temporary_path}/reproj_fuels.tif')
clip_tiff_with_geojson(f'{temporary_path}/reproj_fuels.tif', geojson_path, f'{temporary_path}/fuels.tif')
runcmd(f'rm {temporary_path}/reproj_fuels.tif')

# saving the updated fuels data 
fuels_data = np.squeeze(rasterio.open(f'{temporary_path}/fuels.tif').read())
np.save(f'{binary_maps_path}/fuels_data.npy', fuels_data)
if PRINT:
    print('fuels_data:', fuels_data.shape)

for DAY in days_list:
    data = []

    date_obj = datetime.strptime(str(DAY), '%Y-%m-%d')
    day_of_year = date_obj.timetuple().tm_yday

    # daymet calendars are no leap so we will remove 366th days
    if day_of_year > 365:
        continue

    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    
    # We'll start by adding the cos(rads) and sin(rads)
    rads = (int(day_of_year)-1)*(360/364)*(np.pi / 180)
    data.append(np.cos(rads)*np.ones(fuels_data.shape))
    data.append(np.sin(rads)*np.ones(fuels_data.shape))

    # Fire Danger Potential Indices
    file_path = f'{fire_danger_path}/reproj_{year}{month:02}{day:02}_{year}{month:02}{day:02}.tiff'
    if os.path.exists(file_path):
        clip_tiff_with_geojson(file_path, geojson_path, f'{temporary_path}/clipped_tiff.tif')
        fire_danger_data = np.squeeze(rasterio.open(f'{temporary_path}/clipped_tiff.tif').read())
        runcmd(f'rm {temporary_path}/clipped_tiff.tif')
        if PRINT:
            print('fire_danger_data:', fire_danger_data.shape)

        # -9999.0 and values above 150 are no burn zones
        fire_danger_data = fire_danger_data.astype('float32')
        fire_danger_data[fire_danger_data == -9999.0] = np.nan
        fire_danger_data[fire_danger_data > 200] = np.nan
        data.append(fire_danger_data)
    else:
        continue

    # NDVI from MODIS
    ndvi_files_by_year = [i for i in os.listdir(MODIS_path) if i.startswith(f'MOD13Q1.061__250m_16_days_NDVI_doy{year}') and i.endswith('.tif')]
    doy_by_year=[]
    for i in ndvi_files_by_year:
        i = i.replace(f'MOD13Q1.061__250m_16_days_NDVI_doy{year}', '')
        i = int(i.split('_')[0])
        doy_by_year.append(i)

    closest_doy_idx = find_closest_number(doy_by_year, int(day_of_year))
    closest_tif_name = ndvi_files_by_year[closest_doy_idx]

    if os.path.exists(f'{temporary_path}{closest_tif_name}'):
        closest_ndvi_tif = rasterio.open(f'{temporary_path}{closest_tif_name}')
        closest_ndvi_data = np.squeeze(closest_ndvi_tif.read())
        closest_ndvi_data = closest_ndvi_data.astype('float32')
		# solves all issues with missing data
        closest_ndvi_data[closest_ndvi_data < 0.0] = np.nan
        data.append(closest_ndvi_data)
        if PRINT:
            print('NDVI SHAPE', closest_ndvi_data.shape, closest_tif_name)

    else:
        runcmd(f'rm {temporary_path}/MOD13Q1.061__250m_16*')
        closest_ndvi_tif = f'{MODIS_path}/{closest_tif_name}'
        update_tiff(closest_ndvi_tif, f'{temporary_path}/updated_MODIS.tif')
        clip_tiff_with_geojson(f'{temporary_path}/updated_MODIS.tif', 
                               geojson_path, 
                               f'{temporary_path}/{closest_tif_name}')
        
        runcmd(f'rm {temporary_path}/updated_MODIS.tif')
        closest_ndvi_tif = rasterio.open(f'{temporary_path}/{closest_tif_name}')
        closest_ndvi_data = np.squeeze(closest_ndvi_tif.read())
        closest_ndvi_data = closest_ndvi_data.astype('float32')
        closest_ndvi_data[closest_ndvi_data < 0.0] = np.nan
        data.append(closest_ndvi_data)
        if PRINT:
            print('NDVI SHAPE', closest_ndvi_data.shape, closest_tif_name)

    # Daymet:
    daymet_files = [f'{temporary_path}/daymet_v4_daily_na_{var}_{year}.tif' for var in daymet_vars]

    if do_all_files_exist(daymet_files):
        for daymet_var_file, daymet_var in zip(daymet_files, daymet_vars):
            daymet_var_data = np.squeeze(rasterio.open(f'{temporary_path}/daymet_v4_daily_na_{daymet_var}_{year}.tif').read(day_of_year))
            daymet_var_data = daymet_var_data.astype('float32')
            daymet_var_data[daymet_var_data == -9999.0] = np.nan
            data.append(daymet_var_data)
    else:
        runcmd(f'rm {temporary_path}/daymet_v4_daily_na_*')
        files_needed = [f'{daymet_path}/daymet_v4_daily_na_{var}_{year}.tif' for var in daymet_vars]
        for daymet_var_file, daymet_var in zip(files_needed, daymet_vars):
            update_tiff(daymet_var_file, f'{temporary_path}/updated_daymet_{daymet_var}.tif')
            clip_tiff_with_geojson(f'{temporary_path}/updated_daymet_{daymet_var}.tif', 
                                   geojson_path, 
                                   f'{temporary_path}/daymet_v4_daily_na_{daymet_var}_{year}.tif')
            runcmd(f'rm {temporary_path}/updated_daymet_{daymet_var}.tif')
            daymet_var_data = np.squeeze(rasterio.open(f'{temporary_path}/daymet_v4_daily_na_{daymet_var}_{year}.tif').read(day_of_year))
            daymet_var_data = daymet_var_data.astype('float32')
            daymet_var_data[daymet_var_data == -9999.0] = np.nan
            data.append(daymet_var_data)

    data = np.stack(data, axis=0)
    np.save(f'{binary_maps_path}/{month:02}-{day:02}-{year}.npy', data)
    print(f'FILE: {month:02}-{day:02}-{year}.npy SAVED')
    
end = TM.time()
complete = (end-start)/60
print('COMPLETE TIME ELAPSED:', round(complete, 2), 'MINUTES')
