#!/usr/bin/python3

# mpiexec -n < workers > python3 transformer_training.py

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
from mpi4py import MPI

sys.path.append('/global/homes/r/russ8/Functions')
sys.path.append('/global/homes/r/russ8/fire_danger/')

from DISTRIBUTED_COMPUTING import Slurm_info, retrieve_DL_model
from TF_FUNCTIONS import tf_set_seeds, load_model, check_trial_files
from FIRE_DANGER_FUNCTIONS import *

####################################  PARAMETERS #############################################################

SEED = 123
EPOCHS = (12, 200)
TUNER = 'BAYESIAN'
WINDOW_SIZE = 30
BATCH_SIZE = 384
NUMBER_OF_BAYESIAN_TRIALS = 64
PATIENCE = 20
METRIC = 'MSE'
DEEP_LEARNING_MODEL = 'Transformer'
RUN_TITLE = f'Residual_{DEEP_LEARNING_MODEL}_final_sum'
CARE_IF_ALL_TRIALS_RAN = False
WHICH_MODEL = 'BUILD_HERE' # BEST_TUNED, BUILD_HERE, OTHER
files_path = '/pscratch/sd/r/russ8/fire_danger/data' # no ending /
sub_batch_size = 50000
number_of_features = 16
    
############################ SETTING THE ENVIRONMENT #######################################################

tf_set_seeds(SEED)
slurm_info = Slurm_info()

def random_select(X, y, sub_batch_size, static_vars):

    n = X.shape[0]
    if n > sub_batch_size:
        indices = np.random.choice(n, size=sub_batch_size, replace=False)
        selected_X = X[indices]
        selected_y = y[indices]
        selected_static = static_vars[indices]
    else:
        selected_X = X
        selected_y = y
        selected_static = static_vars

    return selected_X, selected_y, selected_static


def generator(files, batch_size, static_vars):

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
                                                                       sub_batch_size, 
                                                                       static_vars)

            mmap_arr_X = np.concatenate([mmap_arr_X, static_vars_masked], axis = 2).astype('float32')
            splits = np.ceil(mmap_arr_X.shape[0]/batch_size)
            split_X = np.array_split(mmap_arr_X, splits, axis = 0)
            split_y = np.array_split(mmap_arr_y, splits, axis = 0)

            for X, y in zip(split_X, split_y):
                # print(X.shape, y.shape)
                yield X, y

#synchronize all nodes:
comm = MPI.COMM_WORLD
comm.Barrier()
slurm_rank = int(os.environ['SLURM_PROCID'])

if slurm_rank == 0:
    JOB_TITLE = 'chief'
    port_number = 8008
    nodes = slurm_info.nodes
    chief_node = os.environ.get('SLURMD_NODENAME')
    nodes.remove(chief_node)
    chief_node = [chief_node]
    chief_node_json = json.dumps(chief_node)
    
    nodes_with_port_numbers = []
    for idx, i in enumerate(nodes, start = 1):
        p_number = idx*5 + port_number
        val = f'{i}:{p_number}'
        nodes_with_port_numbers.append(val)
    nodes_with_port_numbers_json = json.dumps(nodes_with_port_numbers)
    print('CHIEF NODE:', chief_node[0] + ':8008', '\nWORKER NODES:', nodes)
    
else:
    JOB_TITLE = 'worker'
    nodes_with_port_numbers_json = None
    chief_node_json = None
    
nodes_with_port_numbers_json = comm.bcast(nodes_with_port_numbers_json, root=0)
nodes_with_port_numbers = json.loads(nodes_with_port_numbers_json)

chief_node_json = comm.bcast(chief_node_json, root=0)
chief_node = json.loads(chief_node_json)[0]

# there should be a worker with an index of 0 and the chief 
# gets an index of 0 as well
work_id = slurm_rank - 1
if work_id < 0:
    work_id = 0

os.environ["TF_CONFIG"] = json.dumps({
   "cluster": {
       "chief": [f"{chief_node}:8008"],
       "worker": nodes_with_port_numbers,
   },
  "task": {"type": JOB_TITLE, "index": work_id}
})

# strategy = tf.distribute.MultiWorkerMirroredStrategy()
strategy = tf.distribute.MirroredStrategy()

# Get the rank of the compute node that it is running on
# print(f'RANK: {slurm_rank} MODEL {MODEL}', flush = True)    

# Set the TF_CONFIG environment variable to configure the cluster setting. 
#Enable TF-AMP graph rewrite: 
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
#Enable Automated Mixed Precision: 
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
# tf.config.run_functions_eagerly(True)
os.environ['TF_KERAS'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# tf.debugging.set_log_device_placement(True) # this turns the output into a mess! (might sometimes be useful however)

#############################################################################################################

data_directory = '/pscratch/sd/r/russ8/fire_danger/data'
RESULTS_PATH = f'/pscratch/sd/r/russ8/fire_danger/results/{RUN_TITLE}'

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if slurm_rank == 0:
    print("Number of GPUs Available: ", len(gpus), flush = True)
    print("Number of CPUs Available: ", len(cpus), flush = True)

# I must make the batch size divisible by the number of replicas (GPUs)
# If I try tuning on many CPUs in parallel this should be adjusted!

BATCH_SIZE = BATCH_SIZE*len(gpus)

if slurm_rank == 0:
    print('GPUs:', gpus, flush = True)
    print('CPUs:', cpus, flush = True)

# num_devices = strategy.num_replicas_in_sync
# print(f'Number of devices: {num_devices}', flush = True)

######################### IMPORTING DATA ####################################################
# The order of the features:
# cos_rads, sin_rads, fire_danger_score, NDVI, prcp, srad, swe, tmax, tmin, vp

######################### IMPORTING DATA ###################################################

# Importing data:
static_vars = np.load('/pscratch/sd/r/russ8/fire_danger/data/full_static_vars.npy')
files = produce_npy_files(data_directory)

# This puts the last file of validation to be 06/30/2023 giving
# us an ideal place to view the fire season starting July 1st
del files[-123:]

val_files = files[-147:]
del files[-147:]

train_n = int(len(files)*sub_batch_size)
# print('train_n:', train_n)

val_n = int(len(val_files)*sub_batch_size)
# print('val_n:', val_n)
############################ SETTING UP THE MODEL ###########################################

start = TM.time()

# tf.keras.metrics.MeanAbsoluteError(name='MAE')
# tf.keras.metrics.F1Score(average='macro', name = 'f1_score')
if slurm_rank == 0:
    print(f'BUILDING', DEEP_LEARNING_MODEL, flush = True)
    
METRIC = METRIC.upper()

if strategy:
    with strategy.scope():

        METRICS = [tf.keras.metrics.MeanSquaredError(name='MSE')]

        model = retrieve_DL_model(DEEP_LEARNING_MODEL)        
        model = model(input_shape = (WINDOW_SIZE, number_of_features),
                             optimizer='adam',
                             loss='MSE',
                             sub_batch_size = sub_batch_size,
                             static_vars = static_vars,
                             metrics = METRICS, 
                             momentum=None,
                             strategy = strategy)
else:
    
    METRICS = [tf.keras.metrics.MeanSquaredError(name='MSE')]
    
    model = retrieve_DL_model(DEEP_LEARNING_MODEL)
    model = model(input_shape = (WINDOW_SIZE, number_of_features),
                         optimizer='adam',
                         loss='MSE',
                         sub_batch_size = sub_batch_size,
                         static_vars = static_vars,
                         metrics = METRICS, 
                         momentum=None,
                         strategy = None)

DIRECTION = model.produce_direction()

########################################### CALLBACKS ############################################################

if slurm_rank == 0:
    print('SETTING CALLBACKS', flush = True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{METRIC}', # early stopping doesn't work with f1_score 
    verbose=1,
    patience=PATIENCE,
    mode=DIRECTION,
    restore_best_weights=True)

best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{RESULTS_PATH}/Models/BEST_MODELS/{DEEP_LEARNING_MODEL}_{TUNER}/model.' + '{epoch:02d}.keras',
    save_weights_only=False,
    monitor=f'val_{METRIC}',
    mode=DIRECTION,
    save_best_only=True)

checkpoint_path = f'{RESULTS_PATH}/Models/CHECKPOINTS/{DEEP_LEARNING_MODEL}_{TUNER}/'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{RESULTS_PATH}/Models/CHECKPOINTS/{DEEP_LEARNING_MODEL}_{TUNER}/model.' + '{epoch:02d}.weights.h5',
    save_weights_only=True,
    monitor=f'val_{METRIC}',
    mode=DIRECTION,
    save_best_only=False)

end_loss_equal_to_nan = tf.keras.callbacks.TerminateOnNaN()

reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=f'val_{METRIC}',
    factor=0.1,
    patience=PATIENCE,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

########################################### Check if Tuner worked #######################################################
    
if CARE_IF_ALL_TRIALS_RAN:
    trials_dir = f'{RESULTS_PATH}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}'
    all_trials_available = check_trial_files(trials_dir)
    
    if all_trials_available:
        pass
    else:
        raise('ONE OR MORE TRIALS IS MISSING DATA!!')
    
########################################### MODEL TRAINING ###############################################################
    
# Options for below: # BEST_TUNED, BUILD_HERE, OTHER
if WHICH_MODEL.upper() == 'BEST_TUNED':
    if slurm_rank == 0:
        print(f'BUILDING {DEEP_LEARNING_MODEL} FROM TUNED HYPERPARAMETERS', flush = True)
    
    if TUNER == 'HYPERBAND':
    # Hyperband determines the number of models to train in a bracket by computing 1 + log_factor(max_epochs) and rounding it up to the nearest integer.

        tuner = kt.Hyperband(model,
                         objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION),
                         max_epochs=EPOCHS[1],
                         overwrite=False,
                         factor=3,
                         directory = f'{RESULTS_PATH}/Tuning/',
                         project_name = f'TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}/',
                         distribution_strategy = None)

    elif TUNER == 'BAYESIAN':

        tuner = kt.BayesianOptimization(model,
                         objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION), 
                         max_trials = NUMBER_OF_BAYESIAN_TRIALS,
                         overwrite = False,
                         directory = f'{RESULTS_PATH}/Tuning/',
                         project_name = f'TUNING_{DEEP_LEARNING_MODEL}_Fire_Danger_{TUNER}/',
                         distribution_strategy = None)
        
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] # Get the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    if slurm_rank == 0:
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[-1]
        print()
        print(f"Best trial ID: {best_trial.trial_id}")
        print(best_trial.summary())
        print()
    
elif WHICH_MODEL.upper() == 'BUILD_HERE':
    if slurm_rank == 0:
        print(f'BUILDING {DEEP_LEARNING_MODEL} FROM PARAMETERS YOU PROVIDED ON THIS SCRIPT', flush = True)
        
    parameters = {'batch_size' : BATCH_SIZE,
                  'num_encoder_blocks' : 3,
                  'mlp_layers' : 1,
                  'mlp_units' : 64,
                  'mlp_activation_function' : 'relu',
                  'mlp_dropout' : 0.2,
                  'transformer_stack_dropout' : 0.2,
                  'attention_head_size' : 64, # so this was scaled up by 16: 4*16 = 64 (number_of_heads*16)
                  'number_of_heads' : 4,
                  'feed_forward_dimensions' : 64,
                  'learning_rate' : 0.0001,
                 }
    model = model.build(hp=None, parameters=parameters)
    starting_epoch = 25
    latest_checkpoint = os.path.join(checkpoint_path, f'model.{starting_epoch}.weights.h5')
    
    # I'm adding this manually but this needs to be automated in the future:
    model.load_weights(latest_checkpoint)
    
elif WHICH_MODEL.upper() == 'OTHER':
    path_to_model = '/path/to/model/to/use.keras'
    if slurm_rank == 0:
        print(f'USING THE {DEEP_LEARNING_MODEL} FROM THE PATH YOU PROVIDED', flush = True)
    model = load_model(path_to_model)
    
with strategy.scope():
    
    print('TRAINING IN PROGRESS', flush = True)

    hist = model.fit(generator(files, BATCH_SIZE, static_vars),
                     validation_data = generator(val_files, BATCH_SIZE, static_vars),
                     steps_per_epoch = int(np.ceil(train_n/BATCH_SIZE)),
                     validation_steps = int(np.ceil(val_n/BATCH_SIZE)),
                     initial_epoch = starting_epoch,
                     epochs = EPOCHS[1],
                     batch_size = BATCH_SIZE,
                     callbacks = [early_stopping,
                                  best_model_callback,
                                  checkpoint_callback,
                                  end_loss_equal_to_nan,
                                  reduce_learning_rate])

    val_METRIC_per_epoch = hist.history[f'val_{METRIC}']

    if DIRECTION.upper() == 'MAX':
        best_epoch = val_METRIC_per_epoch.index(max(val_METRIC_per_epoch)) + 1
    elif DIRECTION.upper() == 'MIN':
        best_epoch = val_METRIC_per_epoch.index(min(val_METRIC_per_epoch)) + 1

    print(f'{DEEP_LEARNING_MODEL}: TRAINING COMPLETE BEST EPOCH: {best_epoch}', flush = True)
           
stop = TM.time()
complete = (stop - start)/3600
if slurm_rank == 0:
    print('Process complete! Took ', round(complete, 6), 'hours', flush = True)


