#!/usr/bin/python3
# conda activate DL
# python3 Basic_Transformer.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import time as TM
import random
import pickle
import copy
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
import keras_tuner as kt
import tensorflow_addons as tfa

sys.path.append('../../Functions/')

from STANDARD_FUNCTIONS import write_pickle, read_pickle
from TF_FUNCTIONS import convert_to_TF_data_obj

####################################  PARAMETERS #############################################################
SEED = 123
EPOCHS = (25, 20, 300)
BATCH_SIZE = 3200 # arbitrary
PATIENCE = 5
OPTIMIZER = 'ADAM'
TUNER = 'BAYESIAN'
METRIC = 'MAE'
LOSS = 'MAE'
MOMENTUM = 0.8
TUNE = True
BAYESIAN_TRIALS = 50
PERC_TRAIN = 0.60
PERC_VAL = 0.20
output_path = '/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer'
##############################################################################################################

if TUNE:
	print('ARE YOU SURE YOU WANT TO TUNE? CURRENT SETTINGS WILL DELETE ALL PREVIOUS TRIALS (yes/no)')
	x = input()
	if x.upper() == 'YES':
		pass
	elif x.upper() == 'NO':
		raise Exception('PROJECT HAULTED')
	else:
		raise Exception('PROJECT HAULTED YOU DID NOT SAY YES OR NO')
		
# Checkpointing
checkpoint_path = '/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/Models/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only = True, period = 5)

####################################  FUNCTIONS  ##############################################################

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(SEED)

######################################### ESTABLISHING METRICS #######################################################

MSLE = keras.metrics.MeanSquaredLogarithmicError(name = 'MSLE')
RMSE = keras.metrics.RootMeanSquaredError(name="RMSE", dtype=None)
MAE = tf.keras.metrics.MeanAbsoluteError(name='MAE', dtype=None)
COS_SIM = keras.metrics.CosineSimilarity(name="cosine_similarity", dtype=None, axis=-1)
LOG_COSH_ERROR = keras.metrics.LogCoshError(name="logcosh", dtype=None)
METRICS = [MSLE, RMSE, MAE, COS_SIM, LOG_COSH_ERROR]

if METRIC in ['MSLE', 'RMSE', 'MAE', 'LOG_COSH_ERROR', 'COS_SIM', 'loss']:
	direction = 'min'

if LOSS == 'MSLE':
    LOSS = tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto", name="MSLE")

elif LOSS == 'MSE':
    LOSS = tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.AUTO, name='MSE')

elif LOSS == 'HUBER':
    LOSS = tf.keras.losses.Huber(delta=1.0, name='HUBER')

elif LOSS == 'MAE':
	LOSS = tf.keras.losses.MeanAbsoluteError(name='MAE')

para_relu = PReLU()
leaky_relu = LeakyReLU(alpha=0.01)

#####################################################################################################################

train_X = np.load('/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/final_input/train_X.npy')
train_y = np.load('/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/final_input/train_y.npy')

input_shape = train_X.shape[1:]

val_X = np.load('/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/final_input/val_X.npy')
val_y = np.load('/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/final_input/val_y.npy')

test_X = np.load('/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/final_input/test_X.npy')
test_y = np.load('/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/final_input/test_y.npy')
 
train_data = convert_to_TF_data_obj(train_X, train_y, BATCH_SIZE)
val_data = convert_to_TF_data_obj(val_X, val_y, BATCH_SIZE)
test_data = convert_to_TF_data_obj(test_X, test_y, BATCH_SIZE)

del train_X, train_y, val_X, val_y, test_X, test_y

def build_model(hp, OPT=OPTIMIZER):
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    transformer_blocks = hp.Int("transformer_blocks", 3, 6)
    for j in range(transformer_blocks):
        
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=hp.Int(f'head_size_{j}', min_value=32, max_value=256, step=32), num_heads=hp.Int("number_of_heads", 3, 5), dropout=0)(x, x)
        if hp.Boolean("dropout_1"):
            x = layers.Dropout(0.25)(x)
        else:
            x = layers.Dropout(0)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=4, kernel_size=1, activation=hp.Choice("activation_function_1", ["leaky_relu", 'relu']))(x) # I removed elu
        if hp.Boolean("dropout_2"):
            x = layers.Dropout(0.4)(x)
        else:
            x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = x + res
        
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(hp.Int(f'MLP_Number_of_Neurons', min_value=32, max_value=256, step=32), activation=hp.Choice("activation_function_2", ["leaky_relu", 'relu', 'tanh']))(x) 
    if hp.Boolean("dropout"):
        x = layers.Dropout(0.25)(x)
    else:
        x = layers.Dropout(0)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    hp_learning_rate = [0.1, 1e-2, 1e-3]
	
    if OPT == 'ADAM':
        opt=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values = hp_learning_rate))
    elif OPT == 'SGD':
        opt = tf.keras.optimizers.experimental.SGD(learning_rate=hp.Choice('learning_rate', values = hp_learning_rate), momentum = MOMENTUM)
	
    model.compile(
    loss= LOSS,
	optimizer = opt,
    metrics=METRICS)

    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{METRIC}',
    verbose=1,
    patience=PATIENCE,
    mode=direction,
    restore_best_weights=True)

start = TM.time()

############################### MODEL TUNING (OR REFERENCING THE TUNER) ######################################################

if TUNER == 'HYPERBAND':
# Hyperband determines the number of models to train in a bracket by computing 1 + log_factor(max_epochs) and rounding it up to the nearest integer.

    tuner = kt.Hyperband(build_model,
                         objective = kt.Objective(f'val_{METRIC}', direction=direction),
                         max_epochs=EPOCHS[1],
                         overwrite=False,
                         factor=3,
                         directory = '/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/Tuning',
                         project_name = 'TUNING_BASIC_TRANSFORMER_HYPERBAND')

elif TUNER == 'BAYESIAN':
	
    tuner = kt.BayesianOptimization(build_model,
                         objective = kt.Objective(f'val_{METRIC}', direction=direction), 
                         max_trials = BAYESIAN_TRIALS,
                         overwrite=False,
                         directory = '/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/Tuning',
                         project_name = 'TUNING_BASIC_TRANSFORMER_BAYESIAN')
if TUNE:
	# if you want to tune the model
	print('TUNING IN PROGRESS')
	tuner.overwrite = True
	tf.keras.backend.clear_session()
	tuner.search(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS[0], validation_data=val_data, callbacks=[early_stopping])
	print('TUNING COMPLETE')

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

########################################### MODEL TRAINING ###############################################################

early_stopping.patience = PATIENCE + 20
hist = model.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS[2], validation_data=val_data, callbacks=[early_stopping, cp_callback])
val_METRIC_per_epoch = hist.history['val_loss']
best_epoch = val_METRIC_per_epoch.index(min(val_METRIC_per_epoch)) + 1

print(f'BEST EPOCH: {best_epoch}')
hypermodel = tuner.hypermodel.build(best_hps) # don't comment this out!! Must re-initiate the model
hypermodel.fit(train_data, epochs=best_epoch, validation_data=val_data)
hypermodel.save(f'/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/Models/TRANSFORMER_{TUNER.upper()}') 

predictions = np.squeeze(hypermodel.predict(test_data))

np.save(f'/mnt/locutus/remotesensing/r62/fire_danger/Results_Basic_Transformer/Output/PREDICTIONS_{TUNER.upper()}.npy', predictions)

stop = TM.time()
complete = (stop - start)/3600

print('Process complete! Took ', round(complete, 2), 'hours')
