#!/usr/bin/python3

import os
import numpy as np
import pandas as pd

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

def convert_to_classification(arr, min_arr, max_arr):
    arr = arr * (max_arr - min_arr) + min_arr
    vectorized_conversion = np.vectorize(convert_to_int)
    arr = arr.astype('float32')
    arr = vectorized_conversion(arr)
    arr = arr.astype('int32')
    # arr = tf.keras.utils.to_categorical(arr, num_classes=13)
    
    return np.expand_dims(arr, -1) # needed if I use f1_score because of its design

def produce_npy_files(data_directory):
    files = np.array([i for i in os.listdir(data_directory) if i.startswith('TRANSFORMER_DATA_ALL_')])
    files_to_compare_to = np.array([f'TRANSFORMER_DATA_ALL_{i}.npy' for i in range(29, 565)])
    files = [x for x in files_to_compare_to if x in files]
    return [f'{data_directory}/{file}' for file in files]
    
def generator(files, mask, static_vars, batch_size, classification = False):

    while True:
        for idx, file in enumerate(files):
            if idx == int(len(files) - 1):
                continue

            mmap_arr_X = np.load(file, mmap_mode = 'r')
            mmap_arr_X = mmap_arr_X[~mask]

            mmap_arr_y = np.load(files[idx+1], mmap_mode = 'r')
            mmap_arr_y = mmap_arr_y[~mask]
            mmap_arr_y = mmap_arr_y[:, -1, 2]

            if classification:
                mmap_arr_y = convert_to_classification(mmap_arr_y, min_fire_danger, max_fire_danger)
            
            mmap_arr_X = np.concatenate([mmap_arr_X, static_vars], axis = 2).astype('float32')

            splits = np.ceil(mmap_arr_X.shape[0]/batch_size)
            split_X = np.array_split(mmap_arr_X, splits, axis = 0)
            split_y = np.array_split(mmap_arr_y, splits, axis = 0)

            for X, y in zip(split_X, split_y):
                yield X, y
                # print(X.shape, y.shape, file)