#!/usr/bin/python3
# module load tensorflow/2.15.0

import numpy as np
import pandas as pd
import sys
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # forces it to use cpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEEP_LEARNING_MODEL = 'Transformer'
BATCH_SIZE = 384 #256
forecast_window = 7
number_of_features = 16
model_number_in_checkpoints = 13

RUN_TITLE = f'Residual_{DEEP_LEARNING_MODEL}_final_sum'
data_directory = '/pscratch/sd/r/russ8/fire_danger/data'
results_directory = f'/pscratch/sd/r/russ8/fire_danger/results/{RUN_TITLE}'

sys.path.append('/global/homes/r/russ8/Russ_deep_learning_models/')
from Transformer import *

sys.path.append('/global/homes/r/russ8/Functions')
from TF_FUNCTIONS import load_model

sys.path.append('/global/homes/r/russ8/fire_danger/')
from FIRE_DANGER_FUNCTIONS import produce_npy_files

files = produce_npy_files(data_directory)
full_static_vars = np.load('/pscratch/sd/r/russ8/fire_danger/meta_data/full_static_vars.npy')

# This puts the last file of validation to be 06/30/2023 giving
# us an ideal place to view the fire season starting July 1st.
# I wouldn've selected a larger window than 33 days but there 
# is a month gap from 08/02 - 09/03 

# printing the dates that will be evaluated:
# test_files = files[-123:][:33]
# for f in test_files:
#     print(f[-14:-4])

test_files = files[-123:][:33]
model_path = f'{results_directory}/Models/BEST_MODELS/Transformer_BAYESIAN/model.{model_number_in_checkpoints:02}.keras'
model = load_model(model_path)

# the variable indices except the second one as that 
# is the dependent variable WFPI
all_but_2 = np.arange(number_of_features)
all_but_2 = np.delete(all_but_2, 2)

week_number = rank + 1
starting_index = rank*forecast_window
    
input_sequence = np.load(test_files[starting_index])
input_sequence = np.concatenate([input_sequence, full_static_vars], axis = 2)

future_data = [np.load(f, mmap_mode = 'r') for f in test_files[starting_index + 1 : starting_index + 1 + forecast_window]]
future_data = [np.concatenate([f, full_static_vars], axis = 2) for f in future_data]

actuals = [np.squeeze(a[:, -1, 2]) for a in future_data]

# Create a null model aka use previous fire danger scores to predict next day 
# fire danger scores:
null = np.squeeze(np.load(test_files[starting_index])[:, -1, 2])

predictions = []

for n in range(forecast_window):
    print(f'CALCULATING PREDICTIONS FOR DAY:', n, 'WEEK:', week_number)
    mask = ~(np.isnan(input_sequence).any(axis=(1, 2)))

    preds = np.squeeze(model.predict(input_sequence[mask],
                                     batch_size = BATCH_SIZE))
    next_day_prediction = np.ones(input_sequence.shape[0])*np.nan
    next_day_prediction[mask] = preds

    predictions.append(next_day_prediction)

    input_sequence = np.roll(input_sequence, -1, axis=1)
    input_sequence[:, -1, 2] = next_day_prediction
    input_sequence[:, -1, all_but_2] = future_data[n][:, -1, all_but_2]

actuals = np.stack(actuals)
np.save(f'{results_directory}/Output/actuals_window_{forecast_window}_week_{week_number}.npy', actuals)

predictions = np.stack(predictions)
np.save(f'{results_directory}/Output/predictions_window_{forecast_window}_week_{week_number}.npy', predictions)

np.save(f'{results_directory}/Output/null_week_{week_number}.npy', null)

print(f'PROCESSING FORECASTS FOR WEEK:', week_number, 'COMPLETE')
    