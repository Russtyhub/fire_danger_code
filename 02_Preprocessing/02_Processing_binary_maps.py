#!/usr/bin/python3
# conda activate DL

import numpy as np
import os
import sys
import pandas as pd
from datetime import date
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('/home/r62/repos/russ_repos/Functions/')
from STANDARD_FUNCTIONS import runcmd, format_with_zeros, read_pickle
from TIME import create_list_of_dates

os.chdir('/mnt/locutus/remotesensing/r62/fire_danger/binary_data_maps_V2/')

start_date = date(2020, 1, 1)
end_date = date(2023, 12, 31)

# creating a list of dates to create the ordered list of binary map files
dates_of_analysis = create_list_of_dates(start_date, end_date)
bin_files = []
for DATE in dates_of_analysis:
    Date_str_list = str(DATE).split('-')
    year = Date_str_list[0]
    month = format_with_zeros(int(Date_str_list[1]), 2)
    day = format_with_zeros(int(Date_str_list[2]), 2)
    filename = f'{month}-{day}-{year}.npy'
    bin_files.append(filename)
	
# Columns order: 
# cos_rads, sin_rads, fire_danger_score_data, NDVI, prcp, srad, swe, tmax, tmin, vp

missing_files = []

print('THERE ARE', len(os.listdir()), 'BINARY MAP FILES IN THE DIRECTORY')
mins = []
maxes = []

for bin_map_name in tqdm(bin_files, desc="Processing", unit="iteration"):

	if os.path.isfile(bin_map_name):
		npy_file = np.load(bin_map_name)

		# Recording mins and maxes and saving the results
		# Don't normalize now! we must see if any images are outliers in 
		# terms of their mins and maxes
		mins.append(np.nanmin(npy_file, axis = (1, 2)))
		maxes.append(np.nanmax(npy_file, axis = (1, 2)))

	else:
		missing_files.append(bin_map_name)

bin_files = np.array(bin_files)
missing_files = np.array(missing_files)
files_gathered = np.setdiff1d(bin_files, missing_files)

mins_df = pd.DataFrame(np.stack(mins))
mins_df.columns = ['min_cos_rads', 'min_sin_rads', 'min_fire_danger_score_data',
				   'min_NDVI', 'min_prcp', 'min_srad', 'min_swe', 'min_tmax', 'min_tmin', 'min_vp']
mins_df['filename'] = files_gathered

maxes_df = pd.DataFrame(np.stack(maxes))
maxes_df.columns = ['max_cos_rads', 'max_sin_rads', 'max_fire_danger_score_data',
				   'max_NDVI', 'max_prcp', 'max_srad', 'max_swe', 'max_tmax', 'max_tmin', 'max_vp']
maxes_df['filename'] = files_gathered

mins_df.to_pickle('/mnt/locutus/remotesensing/r62/fire_danger/meta/binary_maps_mins.pkl')
maxes_df.to_pickle('/mnt/locutus/remotesensing/r62/fire_danger/meta/binary_maps_maxes.pkl')

# Plotting mins and maxes for each variable over their respective files

min_cols = ['min_fire_danger_score_data', 'min_NDVI', 'min_prcp', 'min_srad', 'min_swe', 'min_tmax', 'min_tmin', 'min_vp']
max_cols = ['max_fire_danger_score_data', 'max_NDVI', 'max_prcp', 'max_srad', 'max_swe', 'max_tmax', 'max_tmin', 'max_vp']

fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(12, 25))
for idx, (min_col, max_col) in enumerate(zip(min_cols, max_cols)):
    axs[idx, 0].set_title(min_col.replace('_', ' '))
    axs[idx, 0].plot(mins_df[min_col])
    
    axs[idx, 1].set_title(max_col.replace('_', ' '))
    axs[idx, 1].plot(maxes_df[max_col])
	
plt.savefig('/mnt/locutus/remotesensing/r62/fire_danger/images/mins_maxes_over_time.png', dpi=300)

# Normalize the binary maps
mins = np.array(mins_df.iloc[:, :-1].min())
maxes = np.array(maxes_df.iloc[:, :-1].max())

for FILE in tqdm(files_gathered, desc="Processing", unit="iteration"):
	npy_file = np.load(FILE)
	
	# normalizing the arrays
	normalized_data = (npy_file - mins[:, None, None]) / (maxes - mins)[:, None, None]
	np.save(f'/mnt/locutus/remotesensing/r62/fire_danger/normed_maps_V2/{FILE}', normalized_data)


