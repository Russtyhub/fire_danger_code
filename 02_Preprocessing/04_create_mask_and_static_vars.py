#!/usr/bin/python3
# conda activate DL

import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/path/to/functions/')
from STANDARD_FUNCTIONS import find_factors
from DATA_ANALYSIS_FUNCTIONS import Assign_numbers_to_ordinal_vars

CREATE_CUMULATIVE_MASK = True
CREATE_STATIC_VARS = True
EDIT_FUELS_KEY = False
data_path = 'path/to/where/you/are/storing/project/data'
fuels_data = np.load(f'{data_path}/fuels/FINAL_FUELS_F40_DS_MODE.npy')
static_inputs_dir = f'{data_path}/pre_transformer/full_datasets/static_inputs'
data_dir = f'{data_path}/pre_transformer/full_datasets/dynamic_inputs'
meta_dir = f'{data_path}/meta'
WINDOW_SIZE = 30

# edit the fuels data key:
if EDIT_FUELS_KEY:
    Fuel_Data_Key = pd.read_csv(f'{data_path}/fuels/Fuel_Data_Key.csv')
    Fuel_Data_Key.drop(['Class Description', 'Frequency (downsampled using mode)', 'Unnamed: 4'], axis = 1, inplace = True)
    mapping_dict = {'0': 0, 'very low': 1, 'low': 2, 'moderate': 3, 'high': 4, 'very high': 5, 'extreme': 6}

    adjust = Assign_numbers_to_ordinal_vars(Fuel_Data_Key, ['Flame length', 'Rate of spread'], mapping_dict)
    # adjust.show()
    Fuel_Data_Key = adjust.assign_numbers()

    Fuel_Data_Key.loc[23, ['Flame length', 'Rate of spread']] = [5.0, 5.0]
    Fuel_Data_Key.loc[45, ['Rate of spread']] = 5.0

    Fuel_Data_Key['Flame length'] = Fuel_Data_Key['Flame length'].astype('int16')
    Fuel_Data_Key['Rate of spread'] = Fuel_Data_Key['Rate of spread'].astype('int16')
    Fuel_Data_Key.to_csv(f'{data_path}/fuels/Fuels_Key_Editted.csv')

print('How I arrived at a batch size of 929')
number = 121699 # number of values that are always available (main mask)
factors = find_factors(number)
print(f"The factors of {number} are: {factors}")

if CREATE_CUMULATIVE_MASK:
    print('CREATING CUMULATIVE MASK')
    os.chdir(data_dir)

    # Initialize the cumulative mask
    cumulative_mask = np.zeros((956592,), dtype=bool)
    files = [i for i in os.listdir() if i.startswith('TRANSFORMER_DATA_ALL_') and i.endswith('.npy')]

    # Iterate over each mask file
    for file in tqdm(files, desc="Processing mask files"):
        mask = np.load(file)
        mask = np.any(np.isnan(mask), axis=(1, 2))
        cumulative_mask |= mask  # Update the cumulative mask with bitwise OR

    # lastly I account for missing values in the fuels dataset
    cumulative_mask |= np.isnan(fuels_data).flatten()
    # reverse it so the mask represents values that DO exist
    cumulative_mask = ~cumulative_mask

    print('NUMBER OF OBSERVATIONS TO TRAIN WITH FOR EACH DAY:', cumulative_mask.sum())
    test = np.load(f'{data_dir}/TRANSFORMER_DATA_ALL_99_2020-04-09.npy')
    result = np.isnan(test[cumulative_mask]).sum()
    print(f'LOOK AT THIS TEST FILE, THIS SHOULD BE ZERO (0 MISSING) \nAFTER APPLYING THE MASK TO ANY OF OUR FILES: {result}')

    np.save(f'{meta_dir}/cumulative_mask.npy', cumulative_mask)

if CREATE_STATIC_VARS:
    print()
    print('CREATING STATIC VARIABLES TO APPEND TO INPUTS')
    mask = np.load(f'{meta_dir}/cumulative_mask.npy')

    fuels_data = fuels_data.flatten()
    # fuels_data = fuels_data[~mask]

    fuels_data_key = pd.read_csv(f'{data_path}/fuels/Fuels_Key_Editted.csv')
    fuels_data_key.drop(['Unnamed: 0', 'Class Label'], axis = 1, inplace = True)

    for col in fuels_data_key:
        fuels_data_key[col] = fuels_data_key[col].astype('float32')
        
    num_columns = fuels_data_key.iloc[:, 1:].shape[1]
    expanded_array = np.zeros((fuels_data.shape[0], num_columns))

    for i, value in enumerate(fuels_data):
        corresponding_row = fuels_data_key[fuels_data_key['Data Encoding'] == value]
        if not corresponding_row.empty:
            expanded_array[i] = corresponding_row.iloc[0, 1:].values  # Extract the values excluding 'Data Encoding' column
            
    # I need expanded array to be the same for every day of
    # the lookback window (one month)
            
    # Order of the columns of the 2nd dimension of the expanded_array:
    # 'Fine fuel load (t/ac)',
    # 'Characteristic SAV (ft-1)',
    # 'Packing ratio (dimensionless)',
    # 'Extinction moisture content (percent) 25', 
    # 'Flame length',
    # 'Rate of spread'

    static_vars = []
    for _ in range(WINDOW_SIZE):
        static_vars.append(expanded_array)
        
    static_vars = np.stack(static_vars)
    static_vars = np.moveaxis(static_vars, 0, 1)
    del expanded_array

    static_vars_maxes = np.max(static_vars, axis = (0, 1))
    static_vars_mins = np.min(static_vars, axis = (0, 1))

    normed_static_vars = (static_vars - static_vars_mins)/(static_vars_maxes - static_vars_mins)
    np.save(f'{static_inputs_dir}/full_static_vars.npy', normed_static_vars)

    normed_static_vars = normed_static_vars[mask]
    np.save(f'{static_inputs_dir}/static_vars.npy', normed_static_vars)


