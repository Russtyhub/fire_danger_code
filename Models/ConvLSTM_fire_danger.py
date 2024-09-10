#!/usr/bin/python3
# conda activate DL
# python3 ConvLSTM_fire_danger.py

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# import keras_tuner as kt
import random
import datetime
from datetime import date, timedelta
import copy
import time as TM

######## PARAMETERS ############################################################

SUMMIT = False
SEED = 123
LOOKBACK = 30
EPOCHS = 75
BATCH_SIZE = 1
PATIENCE = 5
METRIC = 'MSE'
IMPORT_CHECK = False

################################################################################

# tf.config.run_functions_eagerly(True)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.config.run_functions_eagerly(True)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print("Number of GPUs Available: ", len(gpus), flush = True)
print("Number of CPUs Available: ", len(cpus), flush = True)

# strategy_gpus = tf.distribute.experimental.MultiWorkerMirroredStrategy(
# 	communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
# 	devices = [f"/GPU:{i}" for i in range(len(gpus))])

# strategy_cpus = tf.distribute.experimental.MultiWorkerMirroredStrategy(
# 	communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
# 	devices = [f"/CPU:{i}" for i in range(len(cpus))])

strategy_gpus = tf.distribute.MirroredStrategy(devices = [f"/GPU:{i}" for i in range(len(gpus))])
strategy_cpus = tf.distribute.MirroredStrategy(devices = [f"/CPU:{i}" for i in range(len(cpus))])

if SUMMIT:
	INPUT_DIR = './Input_Data'
	RESULTS_PATH = './Results'
else:
	INPUT_DIR = '/mnt/locutus/remotesensing/r62/fire_danger/normed_bin_data'
	RESULTS_PATH = '/mnt/locutus/remotesensing/r62/fire_danger/Results_convlstm'
	strategy_gpus = strategy_cpus
	
os.chdir(INPUT_DIR)

if os.path.isdir(f'{RESULTS_PATH}/Output/') and os.path.isdir(f'{RESULTS_PATH}/Models/') and os.path.isdir(f'{RESULTS_PATH}/Tuning/'):
	pass

else:
	os.makedirs(f'{RESULTS_PATH}/Output/')
	os.makedirs(f'{RESULTS_PATH}/Models/')
	os.makedirs(f'{RESULTS_PATH}/Tuning/')
	
######## FUNCTIONS #############################################################

def tf_set_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	tf.random.set_seed(seed)
	np.random.seed(seed)

tf_set_seeds(SEED)

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


def convert_to_TF_data_obj(data_X, data_y, BATCH_SIZE):
		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		if data_y.all() != None:
			data = tf.data.Dataset.from_tensor_slices((data_X, data_y))
		else:
			data = tf.data.Dataset.from_tensor_slices((data_X))
		data = data.batch(BATCH_SIZE)
		data = data.with_options(options)
		return data


# with strategy_cpus.scope():

# 	def generator(list_by_area, epochs, lookback):
# 		final = []
# 		for _ in range(epochs):
# 			for timestamp, array_of_files in enumerate(list_by_area, start = 1): # I'm removing 19 frames to make it evenly divisible by the lookback window (30)
# 				data = []
# 				data = np.stack([np.moveaxis(np.load(np_array), 0, -1) for np_array in array_of_files])

# 				if (timestamp%(lookback+2) != 0):
# 					final.append(data)

# 				else:
# 					X = np.moveaxis(np.stack(final)[:-1], 0, 1)
# 					y = np.moveaxis(np.stack(final)[1:], 0, 1)
# 					y = np.expand_dims(y[:, :, :, :, 2], (-1))

# 					for i, j in zip(X, y):
# 						i = np.expand_dims(i, 0)
# 						j = np.expand_dims(j, 0)
# 						# print(i.shape, j.shape)
# 						yield i, j

# 					final = []

def generator(list_by_area, lookback):
	final = []

	for timestamp, array_of_files in enumerate(list_by_area, start=1):
		data = np.stack([np.moveaxis(np.load(np_array), 0, -1) for np_array in array_of_files])

		if (timestamp % (lookback + 2) != 0):
			final.append(data)
		else:
			X = np.moveaxis(np.stack(final)[:-1], 0, 1)
			y = np.moveaxis(np.stack(final)[1:], 0, 1)
			y = np.expand_dims(y[:, :, :, :, 2], (-1))
			final = []
			for i, j in zip(X, y):

				yield i, j

def create_tf_dataset(generator_fn, list_by_area, lookback, batch_size = 1):
	dataset = tf.data.Dataset.from_generator(
		lambda: generator_fn(list_by_area, lookback),
		output_signature=(
			tf.TensorSpec(shape=(None, 128, 128, 10), dtype=tf.float16),  # X
			tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float16),  # y
		)
	)
	
	dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	return dataset


if IMPORT_CHECK:
	raise Exception("JUST CHECKING ENVIRONMENT NOT A SERIOUS RUN")
	
######## CREATING TIME SERIES DATASETS ###############################################

# with strategy_cpus.scope():
	
start_date = date(2021, 6, 15)
end_date = date(2022, 12, 31)
days_list = create_list_of_dates(start_date, end_date, x_days=1)

locations = np.arange(64)
files_in_order = []

for DAY in (days_list):
	for location in locations:
		day_of_month = format_with_zeros(DAY.day, 2)
		month = format_with_zeros(DAY.month, 2)
		year = str(DAY.year)

		if os.path.exists(f'{location}_{month}-{day_of_month}-{year}.npy'):
			files_in_order.append(f'{location}_{month}-{day_of_month}-{year}.npy')
		else:
			pass

num_time_steps = len([i for i in files_in_order if i.startswith(f'{str(0)}_')])
list_by_area = np.split(np.array(files_in_order), num_time_steps)[19:]

train_files = list_by_area[0:int(len(list_by_area)/2)][0:10]
val_files = list_by_area[int(len(list_by_area)/2):int((len(list_by_area)/2)+(len(list_by_area)/4))][0:10]
test_files = list_by_area[int((len(list_by_area)/2)+(len(list_by_area)/4)):]

train_dataset = create_tf_dataset(generator, train_files, 30)
val_dataset = create_tf_dataset(generator, val_files, 30)

del train_files, val_files
	
# 	train_X, train_y = generator(train_files, LOOKBACK)
# 	training = convert_to_TF_data_obj(train_X, train_y, BATCH_SIZE)
# 	del train_X, train_y

# 	val_X, val_y = generator(val_files, LOOKBACK)
# 	validation = convert_to_TF_data_obj(val_X, val_y, BATCH_SIZE)
# 	del val_X, val_y

#######################################################################################

# with strategy_gpus.scope():
early_stopping = keras.callbacks.EarlyStopping(monitor=f"val_{METRIC}", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=f"val_{METRIC}", patience=5)

# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, 128, 128, 10), batch_size = None)
MASK = layers.Masking(mask_value=-1.0)(inp)

# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
	filters=64,
	kernel_size=(5, 5),
	padding="same",
	return_sequences=True,
	activation="relu",)(MASK)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
	filters=64,
	kernel_size=(3, 3),
	padding="same",
	return_sequences=True,
	activation="relu",)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
	filters=64,
	kernel_size=(1, 1),
	padding="same",
	return_sequences=True,
	activation="relu",)(x)
x = layers.Conv3D(
	filters=1, kernel_size=(3, 3, 3), activation="linear", padding="same")(x)

model = keras.models.Model(inp, x)
model.compile(loss=tf.keras.losses.MeanSquaredError(name='MSE'), optimizer=keras.optimizers.Adam(), run_eagerly=True)

########################################### MODEL TRAINING ###############################################################

# with strategy_gpus.scope():
	
start = TM.time()

# hist = model.fit(generator(train_files, EPOCHS, 30),
# 				 steps_per_epoch = 64*len(train_files),
# 				 epochs=EPOCHS,
# 				 validation_data=generator(val_files, EPOCHS, 30),
# 				 validation_steps = 64*len(val_files),
# 				 callbacks=[early_stopping, reduce_lr],
# 				 batch_size = 1,
# 				 verbose = 2)

hist = model.fit(train_dataset,
	 epochs = EPOCHS,
	 validation_data = val_dataset,
	 callbacks=[early_stopping, reduce_lr],
	 batch_size = BATCH_SIZE,
	 verbose = 2)

val_METRIC_per_epoch = hist.history[f'val_{METRIC}']
best_epoch = val_METRIC_per_epoch.index(max(val_METRIC_per_epoch)) + 1 
print(f'BEST EPOCH: {best_epoch}', flush = True)

model = keras.models.Model(inp, x)
model.compile(loss=tf.keras.losses.MeanSquaredError(name='MSE'), optimizer=keras.optimizers.Adam())

model.fit(train_dataset, epochs=best_epoch, validation_data=val_dataset)
model.save(f'{RESULTS_PATH}/Models/CONVLSTM') 

stop = TM.time()
complete = (stop - start)/3600

print('Process complete! Took ', round(complete, 2), 'hours', flush = True)

