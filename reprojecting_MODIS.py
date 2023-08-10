#!/usr/bin/python3
# conda activate r62GEDI

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import date, timedelta
import os
import sys
from datetime import datetime

sys.path.append('../Functions')

from REMOTE_SENSING_FUNCTIONS import clip_tiff_with_geojson
from STANDARD_FUNCTIONS import create_list_of_dates

def runcmd(cmd, verbose = True, *args, **kwargs):

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

os.chdir('/mnt/locutus/remotesensing/r62/fire_danger/appeears_CA')
modis_files = [i for i in os.listdir() if i.endswith('.hdf')]

for f in modis_files:
	start_date = f.split('.')[1][1:]
	start_year = start_date[0:4]
	day_num = start_date[4:]
	start_date = datetime.strptime(start_year + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
	location = f.split('.')[2]
	
	end_date = f.split('.')[4]
	end_year = end_date[0:4]
	day_num = end_date[4:7]
	end_date = datetime.strptime(end_year + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
		
	# print(start_date, end_date, location)
	
	if start_date == '2020-07-27':
		runcmd(f'gdal_translate -of GTiff \'HDF4_EOS:EOS_GRID:\"{f}\":MODIS_Grid_16DAY_250m_500m_VI:250m 16 days NDVI\' NDVI_{start_date}_to_{end_date}_{location}.tif')
		runcmd(f'gdalwarp -t_srs EPSG:4326 NDVI_{start_date}_to_{end_date}_{location}.tif NDVI_{start_date}_to_{end_date}_{location}.tif')
		
# dates = np.sort(np.unique(np.array(dates).astype('datetime64')))
# print(dates)
# print()
# print(dates.shape)
