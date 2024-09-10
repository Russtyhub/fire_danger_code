#!/usr/bin/python3
# conda activate DL
# python3 Building_transformer_dataset.py class_label (or All to include all classes of fuel type)

import os
import sys
from datetime import datetime, timedelta
from datetime import date
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('/home/r62/repos/russ_repos/Functions/')
from STANDARD_FUNCTIONS import runcmd, format_with_zeros, read_pickle
from DATA_ANALYSIS_FUNCTIONS import Assign_numbers_to_ordinal_vars
from TIME import create_list_of_dates

######## PARAMETERS #################################################################################

class_label = 'ALL'
BATCH_SAMPLE = None
start_date = date(2020, 1, 1)
end_date = date(2023, 12, 31)
DIR = '/mnt/locutus/remotesensing/r62/fire_danger/normed_maps_V2/'
OUTPUT_DIR = '/mnt/locutus/remotesensing/r62/fire_danger/pre_transformer/full_datasets/'

# The order of the numpy arrays:
# cos_rads, sin_rads, fire_danger_score_data, NDVI, prcp, srad, swe, tmax, tmin, vp

#####################################################################################################

# importing the fuels data
fuels_data = np.load('/mnt/locutus/remotesensing/r62/fire_danger/fuels/FINAL_FUELS_F40_DS_MODE.npy')

# importing my fuels key:
Fuel_Data_Key = pd.read_csv('/mnt/locutus/remotesensing/r62/fire_danger/fuels/Fuels_Key_Editted.csv')

def create_random_mask(n, num_true_values):
	'''
	n (int): length of mask
	num_true_values (int): number of True's throughout mask
	'''
	result = np.zeros(n, dtype=bool)
	true_indices = np.random.choice(n, num_true_values, replace=False)
	result[true_indices] = True
	return result

if class_label.upper() == 'ALL':
	fuel_mask = None
else:
	# I'm separating the data by only the ENCODING_NUMBER fuel category:
	ENCODING_NUMBER = int(Fuel_Data_Key[Fuel_Data_Key['Class Label'] == class_label]['Data Encoding'])
	fuel_mask = (fuels_data == float(ENCODING_NUMBER))

os.chdir(DIR)

# creatd sorted dates to iterate over:
dates = create_list_of_dates(start_date, end_date)
sorted_files = []
for date in dates:
    new_date = f'{date.month:02}-{date.day:02}-{date.year}.npy'
    sorted_files.append(new_date)

sampled_data = []

# Loop through the sorted files
for idx, file_name in enumerate(tqdm(sorted_files, desc="Processing files")):	
	current_date = datetime.strptime(file_name[:-4], '%m-%d-%Y')
	current_date_str = str(current_date).split(' ')[0]

	# Check if there are at least 30 prior files
	if idx >= 30:
		START_DATE = datetime.strptime(sorted_files[idx - 30][:-4], '%m-%d-%Y')

		# Check if all required files in the lookback window exist
		lookback_window = [START_DATE + timedelta(days=j) for j in range(30)]
		# note: required_files sometimes gets really big (bigger than 30) because
		# there are some instances where there are gaps in the available data of the 
		# time series. This is accounted for below at the next if statement:
		required_files = [date.strftime('%m-%d-%Y.npy') for date in lookback_window]

		if all(os.path.isfile(os.path.join(DIR, file)) for file in required_files):
			window_data = [np.load(FILE) for FILE in required_files]
			window_data = np.stack(window_data)
			window_data = window_data.reshape(window_data.shape[0], window_data.shape[1], window_data.shape[2]*window_data.shape[3])
			window_data = np.moveaxis(window_data, -1, 0)

			# mask out by fuel data
			if fuel_mask:
				window_data = window_data[fuel_mask.reshape(fuel_mask.shape[0]*fuel_mask.shape[1]), :, :]
			
			nan_mask = np.isnan(window_data)
			np.save(f'{OUTPUT_DIR}/masks/TRANSFORMER_DATA_{class_label}_{idx}_{current_date_str}.npy', nan_mask)
			
			# Sampling from those stacks of windows (tracks) 
			# because selecting from this fuel type resulted in so many observations
			# (~20K per day) I decided to downsample them to BATCH_SAMPLE per day for easier computing
			# this is for a random mask right now I'm using only specific fuel type
			if BATCH_SAMPLE:
				mask = create_random_mask(n=window_data.shape[0], num_true_values=BATCH_SAMPLE)
				window_data = window_data[mask, :, :]

			np.save(f'{OUTPUT_DIR}/dynamic_inputs/TRANSFORMER_DATA_{class_label}_{idx}_{current_date_str}.npy', window_data)
		else:
			pass
	else:
		pass


