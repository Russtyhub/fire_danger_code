#!/usr/bin/python3

# Perlmutter: module load tensorflow/2.15.0
# python3 transformer_tuning.py

# info about base tuner in Keras:
# https://keras.io/api/keras_tuner/tuners/base_tuner/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import pandas as pd
import time as TM
import sys
import keras_tuner as kt
import json
import random 
import copy

sys.path.append('/global/homes/r/russ8/Functions')

from DISTRIBUTED_COMPUTING import Slurm_info, retrieve_DL_model
from TF_FUNCTIONS import tf_set_seeds, convert_to_TF_data_obj, make_keras_tuner_trials_paths
from STANDARD_FUNCTIONS import create_directory, delete_everything_in_directory
from FIRE_DANGER_FUNCTIONS import *

####################################  PARAMETERS #############################################################

SEED = 123
EPOCHS = (25, 12)
WINDOW_SIZE = 30
# BATCH_SIZE = 929
TUNER = 'BAYESIAN'
NUMBER_OF_BAYESIAN_TRIALS = 64
OVERWRITE_TRIALS = False
PATIENCE = 5
METRIC = 'MSE'
DEEP_LEARNING_MODEL = 'Transformer'
sub_batch_size = 50000
number_of_features = 16
RUN_TITLE = f'Residual_{DEEP_LEARNING_MODEL}_final'

############################ SETTING THE ENVIRONMENT #######################################################

tf_set_seeds(SEED)
slurm_info = Slurm_info()
slurm_rank = int(os.environ['SLURM_PROCID'])    

if slurm_rank == 0:
    os.environ['KERASTUNER_TUNER_ID'] = 'chief'
else:
    worker_id = slurm_rank - 1
    os.environ['KERASTUNER_TUNER_ID'] = f'tuner{worker_id}'

print(os.environ['KERASTUNER_TUNER_ID'], flush = True)
print('RANK:', slurm_rank, flush = True)
print()

#Enable TF-AMP graph rewrite: 
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
#Enable Automated Mixed Precision: 
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
# tf.config.run_functions_eagerly(True)
os.environ['TF_KERAS'] = "1"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# tf.debugging.set_log_device_placement(True) # this turns the output into a mess! (might sometimes be useful however)

data_directory = '/pscratch/sd/r/russ8/fire_danger/data'
RESULTS_PATH = f'/pscratch/sd/r/russ8/fire_danger/results/{RUN_TITLE}'

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print("Number of GPUs Available: ", len(gpus), flush = True)
print("Number of CPUs Available: ", len(cpus), flush = True)

# I must make the batch size divisible by the number of replicas (GPUs)
# If I try tuning on many CPUs in parallel this should be adjusted!
GLOBAL_BATCH_SIZE = BATCH_SIZE*len(gpus)

print('GPUs:', gpus, flush = True)
print('CPUs:', cpus, flush = True)

strategy = tf.distribute.MirroredStrategy()
# strategy = None

# num_devices = strategy.num_replicas_in_sync
# print(f'Number of devices: {num_devices}', flush = True)

paths_to_create = [f'{RESULTS_PATH}/Output/',
                   f'{RESULTS_PATH}/Models/',
                   f'{RESULTS_PATH}/Tuning/',
                   f'{RESULTS_PATH}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}/',
                   f'{RESULTS_PATH}/Models/CHECKPOINTS/']

# It helps to have the trial dirs created ahead of time for keras tuner in parallel

if OVERWRITE_TRIALS:
    tf.keras.backend.clear_session()
    delete_everything_in_directory(f'{RESULTS_PATH}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}/')

trials_paths = make_keras_tuner_trials_paths(number_of_trials = NUMBER_OF_BAYESIAN_TRIALS,
                                             path = f'{RESULTS_PATH}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}')
    
paths_to_create.extend(trials_paths)
create_directory(paths_to_create, PRINT = False)
    
######################### IMPORTING DATA ###################################################

# Importing data:
static_vars = np.load('/pscratch/sd/r/russ8/fire_danger/data/full_static_vars.npy')
files = produce_npy_files(data_directory)

# This puts the last file of validation to be 06/30/2023 giving
# us an ideal place to view the fire season starting July 1st
del files[-123:]

val_files = files[-147:]
del files[-147:]

# train_n = int(len(files)*sub_batch)
# print('train_n:', train_n)

# val_n = int(len(val_files)*sub_batch)
# print('val_n:', val_n)

############################ SETTING UP THE MODEL ###########################################

start = TM.time()

if strategy:
    with strategy.scope():

        METRIC = METRIC.upper()
        METRICS = [tf.keras.metrics.MeanSquaredError(name='MSE')]

        model = retrieve_DL_model(DEEP_LEARNING_MODEL)        
        model = model(input_shape = (WINDOW_SIZE, number_of_features),
                             # batch_size = BATCH_SIZE,
                             sub_batch_size = sub_batch_size,
                             static_vars = static_vars,
                             optimizer='adam',
                             loss='MSE',
                             metrics = METRICS, 
                             momentum=None,
                             strategy = strategy)
else:
    
    METRIC = METRIC.upper()
    METRICS = [tf.keras.metrics.MeanSquaredError(name='MSE')]

    model = retrieve_DL_model(DEEP_LEARNING_MODEL)
    model = model(input_shape = (WINDOW_SIZE, number_of_features),
                         # batch_size = BATCH_SIZE,
                         sub_batch_size = sub_batch_size,
                         static_vars = static_vars,
                         optimizer='adam',
                         loss='MSE',
                         metrics = METRICS, 
                         momentum=None,
                         strategy = None)

DIRECTION = model.produce_direction()

########################################### CALLBACKS ############################################################

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{METRIC}', # early stopping doesn't work with f1_score 
    verbose=1,
    patience=PATIENCE,
    mode=DIRECTION,
    restore_best_weights=True)

end_loss_equal_to_nan = tf.keras.callbacks.TerminateOnNaN()

############################### MODEL TUNING #######################################################################

if TUNER == 'HYPERBAND':
# Hyperband determines the number of models to train in a bracket by computing 
# 1 + log_factor(max_epochs) and rounding it up to the nearest integer.

    tuner = kt.Hyperband(model,
                     objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION),
                     max_epochs = EPOCHS[1],
                     overwrite = False,
                     factor = 3,
                     directory = f'{RESULTS_PATH}/Tuning/',
                     project_name = f'TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}/',
                     distribution_strategy = strategy)

elif TUNER == 'BAYESIAN':
    
    tuner = kt.BayesianOptimization(model,
                     objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION), 
                     max_trials = NUMBER_OF_BAYESIAN_TRIALS,
                     overwrite = False,
                     directory = f'{RESULTS_PATH}/Tuning/',
                     project_name = f'TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}/',
                     distribution_strategy = strategy)
        
print('TUNING IN PROGRESS', flush = True)

tuner.search(training_data = files, # generator(files, mask, static_vars, batch_size=GLOBAL_BATCH_SIZE)
         validation_data = val_files, # generator(val_files, mask, static_vars, batch_size=GLOBAL_BATCH_SIZE)
         # steps_per_epoch = int(np.ceil(train_n/GLOBAL_BATCH_SIZE)),
         # validation_steps = int(np.ceil(val_n/GLOBAL_BATCH_SIZE)),
         # batch_size = GLOBAL_BATCH_SIZE,
         epochs = EPOCHS[0],
         callbacks = [early_stopping, end_loss_equal_to_nan])

print('TUNING COMPLETE', flush = True)

stop = TM.time()
complete = (stop - start)/3600

if slurm_rank == 0:
    print('Process complete! Took ', round(complete, 2), 'hours', flush = True)




