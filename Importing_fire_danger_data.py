#!/usr/bin/python3
# conda activate r62GEDI


# Data for this project comes form the following sources: Appears (NDVI), Daymet (meterological data), (fuel), fire danger  

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import date, timedelta
import os
import sys
sys.path.append('../Functions')

from REMOTE_SENSING_FUNCTIONS import clip_tiff_with_geojson
from STANDARD_FUNCTIONS import create_list_of_dates

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

start_date = date(2020, 1, 1)
end_date = date(2022, 12, 31)

dates = create_list_of_dates(start_date, end_date, 1)

dates_str = []
for i in dates:
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
	
data_path = '/mnt/locutus/remotesensing/r62/fire_danger/fire_danger_scores/'
front_of_url = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/firedanger/download-tool/source_rasters/wfpi-forecast-1/'
front_of_file = 'emodis-wfpi-forecast-1_data_'
cal_geojson = "/mnt/locutus/remotesensing/r62/fire_danger/California_State_Boundary/California_State_Boundary.geojson"

for file_time_stamp in dates_str:

	if os.path.exists(f'{data_path}reproj_{file_time_stamp}.tiff'):
		print('FILE:', file_time_stamp, 'ALREADY EXISTS')

	else:

		try:
			runcmd(f"wget --directory-prefix={data_path} {front_of_url}{front_of_file}{file_time_stamp}.zip", verbose = False)
			runcmd(f"unzip {data_path}{front_of_file}{file_time_stamp}.zip -d {data_path}", verbose = False)
			runcmd(f"rm {data_path}{front_of_file}{file_time_stamp}.zip {data_path}*.xml", verbose = False)

			# clip the tiff to match California geojson file:
			clip_tiff_with_geojson(f'{data_path}emodis-wfpi_data_{file_time_stamp}.tiff', cal_geojson, f'{data_path}emodis-wfpi_data_{file_time_stamp}.tiff')

			runcmd(f'gdalwarp -t_srs EPSG:4326 {data_path}emodis-wfpi_data_{file_time_stamp}.tiff {data_path}reproj_{file_time_stamp}.tiff')
			runcmd(f" rm {data_path}emodis-wfpi_data_{file_time_stamp}.tiff", verbose = False)

			# np.save(f'/mnt/locutus/remotesensing/r62/fire_danger/binary_data/{file_time_stamp}.npy', np.squeeze(rasterio.open(f'{data_path}reproj_{file_time_stamp}.tiff').read()))
		except Exception:
			print('ERROR IMPORTING FILE:', file_time_stamp)


