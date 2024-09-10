#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from datetime import date
import re

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

def convert_to_int(x):
    if x <= 10:
        return 0
    elif 11 <= x <= 20:
        return 1
    elif 21 <= x <= 30:
        return 2
    elif 31 <= x <= 40:
        return 3
    elif 41 <= x <= 50:
        return 4
    elif 51 <= x <= 60:
        return 5
    elif 61 <= x <= 70:
        return 6
    elif 71 <= x <= 80:
        return 7
    elif 81 <= x <= 90:
        return 8
    elif 91 <= x <= 100:
        return 9
    elif 101 <= x <= 120:
        return 10
    elif 121 <= x <= 140:
        return 11
    elif x >= 141:
        return 12

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
    
def convert_to_classification(arr, min_arr, max_arr):
    arr = arr * (max_arr - min_arr) + min_arr
    vectorized_conversion = np.vectorize(convert_to_int)
    arr = arr.astype('float32')
    arr = vectorized_conversion(arr)
    arr = arr.astype('int32')
    # arr = tf.keras.utils.to_categorical(arr, num_classes=13)
    
    return np.expand_dims(arr, -1) # needed if I use f1_score because of its design


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

# if classification:
# mmap_arr_y = convert_to_classification(mmap_arr_y, min_fire_danger, max_fire_danger)
# mmap_arr_X = np.concatenate([mmap_arr_X, static_vars], axis = 2).astype('float32')
# static_vars, classification = False, output_y = True
    
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
                
# old version
# def generator(files, mask, static_vars, batch_size):

#     while True:
#         for idx, file in enumerate(files):
#             if idx == int(len(files) - 1):
#                 continue

#             mmap_arr_X = np.load(file, mmap_mode = 'r')
#             mmap_arr_X = mmap_arr_X[mask]
#             mmap_arr_X = mmap_arr_X.astype('float32')
            
#             mmap_arr_y = np.load(files[idx+1], mmap_mode = 'r')
#             mmap_arr_y = mmap_arr_y[mask]
#             mmap_arr_y = mmap_arr_y[:, -1, 2].astype('float32')
            
#             mmap_arr_X = np.concatenate([mmap_arr_X, static_vars], axis = 2).astype('float32')
#             splits = np.ceil(mmap_arr_X.shape[0]/batch_size)
#             split_X = np.array_split(mmap_arr_X, splits, axis = 0)
#             split_y = np.array_split(mmap_arr_y, splits, axis = 0)

#             for X, y in zip(split_X, split_y):
#                 yield X, y
