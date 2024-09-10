#!/usr/bin/python3
# conda activate r62GEDI

# Data for this project comes from the following sources: Appears (NDVI), Daymet (meterological data), (fuel), fire danger  

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import date, datetime
import os
import sys
path_to_functions_dir = '/home/r62/repos/russ_repos/Functions'
sys.path.append(path_to_functions_dir)

from REMOTE_SENSING_FUNCTIONS import clip_tiff_with_geojson
from STANDARD_FUNCTIONS import runcmd
from TIME import create_list_of_dates

###########################################################

data_path = '/mnt/locutus/remotesensing/r62/fire_danger/fire_danger_V2/'
cal_geojson = "/mnt/locutus/remotesensing/r62/fire_danger/California_State_Boundary/California_State_Boundary.geojson"

OVERWRITE = True
start_date = date(2020, 1, 1)
end_date = date(2023, 12, 31)

###########################################################

dates = create_list_of_dates(start_date, end_date, 1)

failed_files, days_of_week, dates_str = [], [], []

for i in dates:

	date_as_datetime = datetime.strptime(str(i), '%Y-%m-%d')
	day_of_week = date_as_datetime.weekday()
	days_of_week.append(day_of_week)

	yr = i.year
	if int(i.month) < 10:
		month = '0' + str(i.month)
	else:
		month = str(i.month)

	if int(i.day) < 10:
		day = '0' + str(i.day)
	else:
		day = str(i.day)    

	dates_str.append(f'{yr}{month}{day}_{yr}{month}{day}')
	
# days of week are from 0 - 6 but the website is 1 - 7
days_of_week = np.array(days_of_week)
days_of_week = days_of_week + 1

#for file_time_stamp, dow in zip(dates_str, days_of_week):

forecast = 1
front_of_url = f'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/firedanger/download-tool/source_rasters/wfpi-forecast-{forecast}/'
front_of_file = f'emodis-wfpi-forecast-{forecast}_data_'

for file_time_stamp in dates_str: 

	if os.path.exists(f'{data_path}reproj_{file_time_stamp}.tiff') and OVERWRITE == False:
		print('FILE:', file_time_stamp, 'ALREADY EXISTS')
		continue

	try:
		runcmd(f"wget --directory-prefix={data_path} {front_of_url}{front_of_file}{file_time_stamp}.zip", verbose = False)
		runcmd(f"unzip {data_path}{front_of_file}{file_time_stamp}.zip -d {data_path}", verbose = False)
		runcmd(f"rm {data_path}{front_of_file}{file_time_stamp}.zip {data_path}*.xml", verbose = False)

		# clip the tiff to match California geojson file:
		clip_tiff_with_geojson(f'{data_path}emodis-wfpi_data_{file_time_stamp}.tiff', cal_geojson, f'{data_path}emodis-wfpi_data_{file_time_stamp}.tiff')

		runcmd(f'gdalwarp -t_srs EPSG:4326 {data_path}emodis-wfpi_data_{file_time_stamp}.tiff {data_path}reproj_{file_time_stamp}.tiff')
		runcmd(f" rm {data_path}emodis-wfpi_data_{file_time_stamp}.tiff", verbose = False)
		print('FILE', file_time_stamp, 'SUCCESFULLY IMPORTED')

		# np.save(f'/mnt/locutus/remotesensing/r62/fire_danger/binary_data/{file_time_stamp}.npy', np.squeeze(rasterio.open(f'{data_path}reproj_{file_time_stamp}.tiff').read()))
		
	except Exception:
		print('ERROR IMPORTING FILE:', file_time_stamp)
		failed_files.append(file_time_stamp)

# Keeping record of fire danger values that did not import correctly
failed_files = np.array(failed_files)
np.save('/mnt/locutus/remotesensing/r62/fire_danger/meta/unavailable_fire_danger_files.npy', failed_files)

