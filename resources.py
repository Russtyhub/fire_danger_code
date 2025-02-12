#!/usr/bin/python3

import os
import sys
import rasterio
import geopandas as gpd
import pickle
import pandas as pd
import numpy as np

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
		    
		    
		    
def create_list_of_dates(start_date, end_date, x_days=1):
	'''Creates a daily list between two dates.
	Counts by x_days so if x_days = 1 then this will return a 
	list counting one day at a time. If x_days = 7 it will be every
	week after the start_date (might not land on the end_date unless it
	is easily divisible.

	start_date and end_date should be the form: datetime.date(2020, 1, 1)
	x_days: integer
	'''
	dates = []
	delta = end_date - start_date   # returns timedelta

	for i in range(0, delta.days + 1, x_days):
		day = start_date + timedelta(days=i)
		dates.append(day)
	return dates
    
    
def do_all_files_exist(files_list):
	mask = []
	for i in files_list:
		mask.append(os.path.exists(i))
	mask = np.array(mask)

	if 1 == np.sum(mask)/len(mask):
		output=True
	else:
		output=False
	return output
	
	
def format_with_zeros(number, length):
	''' takes an integer and pads with zeros. For example:
	if number = 4 and length = 3 output '004'. If number = 123
	and length = 3 output '123'.'''
	number_str = str(number)
	if len(number_str) >= length:
		return number_str
	else:
		zeros_to_add = length - len(number_str)
		formatted_str = "0" * zeros_to_add + number_str
		return formatted_str
		
		
def read_pickle(filepath):
	with open(filepath, 'rb') as handle:
		return pickle.load(handle)
		
def find_factors(number):
	if number <= 0:
		raise ValueError("The number must be a positive integer.")

	factors = []
	for i in range(1, int(number**0.5) + 1):
		if number % i == 0:
		    factors.append(i)
		    if i != number // i:
		        factors.append(number // i)
	factors.sort()
	return factors
		
class Assign_numbers_to_ordinal_vars():
	def __init__(self, df, ordinal_cols, map):
		self.df = df
		self.ordinal_cols = ordinal_cols
		self.map = map

	def lower(self, x):
		return x.lower().replace('\t', ' ').lstrip(' ').lstrip(' ')

	def show(self):
		print(20*'*')
		for col in self.ordinal_cols:
		    self.df[col] = self.df[col].astype('str')
		    print(self.df[col].apply(self.lower).unique())
		    print(20*'*')

	def assign_numbers(self):
		for col in self.ordinal_cols:
		    self.df[col] = self.df[col].astype('str')
		    self.df[col] = self.df[col].apply(self.lower)
		    self.df[col] = self.df[col].map(self.map)
		return self.df
		
		
class Slurm_info():

	def __init__(self):
		self.slurms_node_list = os.environ.get('SLURM_JOB_NODELIST')
		self.nodes = self.get_nodes_list()
		
	def count_leading_zeros(self, s):
		count = 0
		for char in s:
		    if char == '0':
		        count += 1
		    else:
		        break
		return count

	def get_nodes_list(self):
		if not self.slurms_node_list or '[' not in self.slurms_node_list:
		    return [self.slurms_node_list] if self.slurms_node_list else []
		else:
		    string_split = self.slurms_node_list.split('[')
		    machine = string_split[0]
		    node_numbers = string_split[1].replace(']', '')
		    node_numbers = node_numbers.split(',')
		    nodes = []

		    for n in node_numbers:
		        if '-' in n:
		            vals = n.split('-')
		            pad_n_zeros = self.count_leading_zeros(vals[0])
		            padding = '0' * pad_n_zeros
		            MIN = int(vals[0])
		            MAX = int(vals[1])
		            nodes.extend([machine + padding + str(i) for i in range(MIN, MAX + 1)])
		        else:
		            nodes.append(f'{machine}{n}')      

		    return nodes
        

	
def check_trial_files(directory, remove_missing = False):
	missing_files = []
	for subdir in os.listdir(directory):
		if subdir.startswith("trial_"):
		    subdir_path = os.path.join(directory, subdir)
		    if os.path.isdir(subdir_path):
		        trial_file_path = os.path.join(subdir_path, 'trial.json')
		        if not os.path.isfile(trial_file_path):
		            missing_files.append(subdir_path)

	if not missing_files:
		print("All 'trial' directories contain a 'trial.json' file", flush = True)
		return True
	else:
		print("The following 'trial' directories are missing 'trial.json' files:", flush = True)
		for missing in missing_files:
		    print(missing)
		return False
		

def random_select(X, y, static_vars, select_size):

	n = X.shape[0]
	if n > select_size:
		indices = np.random.choice(n, size=select_size, replace=False)
		selected_X = X[indices]
		selected_y = y[indices]
		selected_static = static_vars[indices]
	else:
		selected_X = X
		selected_y = y
		selected_static = static_vars

	return selected_X, selected_y, selected_static

		
def produce_npy_files(files_path):
    
    files = [f for f in os.listdir(files_path) if f.startswith('TRANSFORMER_DATA_ALL_') and f.endswith('.npy')]
    # Regex pattern to extract the number from the file name
    pattern = r"TRANSFORMER_DATA_ALL_(\d+)_(\d{4}-\d{2}-\d{2})\.npy"

    # Create a list of tuples with (number, file)
    files_with_numbers = []
    for file in files:
        match = re.match(pattern, file)
        if match:
            number = int(match.group(1))
            files_with_numbers.append((number, file))

    # Sort the list of tuples by the number
    files_with_numbers.sort(key=lambda x: x[0])

    # Extract the sorted files
    sorted_files = [f'{files_path}/{file}' for _, file in files_with_numbers]

    return sorted_files
    
def generator(files, static_vars, batch_size, sub_batch):

	mask3 = ~np.any(np.isnan(static_vars), axis=(1, 2))

	while True:
		for idx, file in enumerate(files):
		    if idx == int(len(files) - 1):
		        continue
		        
		    mmap_arr_X = np.load(file, mmap_mode = 'r')
		    mmap_arr_X = mmap_arr_X.astype('float32')
		    mask1 = ~np.any(np.isnan(mmap_arr_X), axis=(1, 2))

		    mmap_arr_y = np.load(files[idx+1], mmap_mode = 'r')
		    mmap_arr_y = mmap_arr_y[:, -1, 2].astype('float32')
		    mask2 = ~np.isnan(mmap_arr_y)

		    mask = mask1*mask2*mask3

		    mmap_arr_X = mmap_arr_X[mask]         
		    mmap_arr_y = mmap_arr_y[mask]
		    static_vars_masked = static_vars[mask]

		    mmap_arr_X, mmap_arr_y, static_vars_masked = random_select(mmap_arr_X, 
		                                                        mmap_arr_y,
		                                                        static_vars_masked,
		                                                        sub_batch)

		    mmap_arr_X = np.concatenate([mmap_arr_X, static_vars_masked], axis = 2).astype('float32')
		    splits = np.ceil(mmap_arr_X.shape[0]/batch_size)
		    split_X = np.array_split(mmap_arr_X, splits, axis = 0)
		    split_y = np.array_split(mmap_arr_y, splits, axis = 0)

		    for X, y in zip(split_X, split_y):
		        # print(X.shape, y.shape)
		        yield X, y
		
		
def make_keras_tuner_trials_paths(number_of_trials, path):

	'''
	If you have more than 999 trials for HP tuning then 
	add to the code and question your decisions
	'''

	paths_to_create = []
	if number_of_trials >= 100:
		for trial in range(number_of_trials):
		    trial_path = f'{path}/trial_{trial:03d}'
		    paths_to_create.append(trial_path)

	elif (number_of_trials <= 99) and (number_of_trials >= 10):
		for trial in range(number_of_trials):
		    trial_path = f'{path}/trial_{trial:02d}'
		    paths_to_create.append(trial_path)

	elif number_of_trials < 10:
		for trial in range(number_of_trials):
		    trial_path = f'{path}/trial_{trial:01d}'
		    paths_to_create.append(trial_path)
		    
	return paths_to_create
		
		
def create_directory(directory_paths, PRINT = True):
	"""
	Check if a directory exists, and create it if it doesn't.
	directory_paths must be a list of strings
	Parameters:
	- directory_paths (list): The path of the directory to check/create.
	"""

	if isinstance(directory_paths, list):

		for DIR in directory_paths:
			if not os.path.exists(DIR):
				os.makedirs(DIR)
				if PRINT:
					print(f"Directory '{DIR}' created.")
			else:
				if PRINT:
					print(f"Directory '{DIR}' already exists.")
				else:
					pass
	else:
		print('directory_paths should be a list type object')

def delete_everything_in_directory(dir_path, verbose=False):
	if not os.path.exists(dir_path):
		print(f"Directory {dir_path} does not exist.")
		return

	for root, dirs, files in os.walk(dir_path, topdown=False):
		# Delete files
		for name in files:
		    file_path = os.path.join(root, name)
		    try:
		        os.remove(file_path)
		        if verbose:
		            print(f"Deleted file: {file_path}", flush = True)
		    except Exception as e:
		        if verbose:
		            print(f"Error deleting file {file_path}: {e}", flush = True)
		        else:
		            pass

		# Delete directories
		for name in dirs:
		    dir_path = os.path.join(root, name)
		    try:
		        shutil.rmtree(dir_path)
		        if verbose:
		            print(f"Deleted directory: {dir_path}", flush = True)
		    except Exception as e:
		        if verbose:
		            print(f"Error deleting directory {dir_path}: {e}", flush = True)
		        else:
		            pass

	# Optionally, delete the root directory itself
	try:
		os.rmdir(dir_path)
		if verbose:
		    print(f"Deleted root directory: {dir_path}", flush = True)
	except Exception as e:
		if verbose:
		    print(f"Error deleting root directory {dir_path}: {e}", flush = True)
		else:
		    pass
		
		
		
		
		
		
		
		
		
		
		
		
		
		
    
