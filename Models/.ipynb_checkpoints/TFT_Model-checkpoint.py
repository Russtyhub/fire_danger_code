#!/usr/bin/python3
# conda activate pytorch
# python3 TFT_Model.py

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import sys
import matplotlib.pyplot as plt
# import tensorflow as tf 
# import tensorboard as tb 
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

sys.path.append('../../Functions')
from STANDARD_FUNCTIONS import read_pickle, write_pickle
############################################## PARAMETERS ################################################

TUNE = True
output_dir = '/mnt/locutus/remotesensing/r62/fire_danger/Results'
batch_size = 16

###########################################################################################################

def zip_model(path_with_versions, output_path):
    vers_list = os.listdir(path_with_versions)

    versions = []
    for v in vers_list:
        if v.startswith('version'):
            versions.append(int(v.split('_')[1]))
        else:
            pass
    versions = np.array(versions)
    best = 'version_' + str(versions.max())

    subprocess_command = 'zip -r ' + output_path + ' ' + path_with_versions + best + '/*'

    p = subprocess.Popen(subprocess_command, stdout=subprocess.PIPE, shell=True) 
    p.communicate()
		
if os.path.exists(output_dir):
	pass
else:
	os.makedirs(f'{output_dir}/Output')
	os.makedirs(f'{output_dir}/Models')
	os.makedirs(f'{output_dir}/Tuning/OPTUNA_STUDY')

DF = pd.read_pickle(f'{output_dir}/TFT_DATASET.pkl')

# Downsampling for experiments
sample_idxs = np.random.choice(np.array(DF['IDX'].unique()), size=500, replace=False)
DF = DF[DF['IDX'].isin(sample_idxs)]

DATA_PKL = {}
DF['Day_Number'] = DF['Day_Number'].astype('int64')
DF['IDX'] = DF['IDX'].astype('int64')
DF['IDX'] = DF['IDX'].astype('object')

DF_train = DF[DF.Day_Number <= 425] # The last 20% is being ignored and will be used later during evaluations (test data)
DF_train.reset_index(inplace=True, drop=True)

actuals_test = np.array(DF[DF.Day_Number > 425]['danger_score'])
DATA_PKL['Actuals_Test'] = actuals_test

max_prediction_length = 106
del actuals_test

max_encoder_length = DF_train['Day_Number'].max()+1
training_cutoff = DF_train["Day_Number"].max() - max_prediction_length

# print('DF_train:\n', DF_train)
print('max_prediction_length:', max_prediction_length)
print('training_cutoff:', training_cutoff)

print('CREATING TRAINING DATASET')

print(DF_train.dtypes)

training = TimeSeriesDataSet(
	DF_train[lambda x: x.Day_Number <= training_cutoff],
	time_idx="Day_Number",
	target="danger_score",
	group_ids=["IDX"],
	min_encoder_length=max_prediction_length, 
	max_encoder_length=max_encoder_length,
	min_prediction_length=1,
	max_prediction_length=max_prediction_length,
	static_categoricals=["IDX"], 
	static_reals = ['fuel'],
	time_varying_known_reals=["Day_Number", 'ndvi', 'swe', 'prcp', 'srad', 'tmax', 'tmin', 'vp', 'cos_rads', 'sin_rads'],
	time_varying_unknown_reals=['danger_score'],
	target_normalizer=GroupNormalizer(groups=["IDX"], transformation="softplus"),  # we normalize by group
	add_relative_time_idx=True,
	add_target_scales=True,
	add_encoder_length=True,
)

print('CREATING VALIDATION DATASET')

validation = TimeSeriesDataSet.from_dataset(training, DF_train, predict=True, stop_randomization=True)

# create dataloaders for  our model
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=6)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*5, num_workers=6)

# Baseline Model
baseline_actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader, return_y=True)


print('BASELINE MODEL ACTUALS VS PREDICTED BASED ON VALIDATION DATASET', (baseline_actuals - baseline_predictions).abs().mean().item())

DATA_PKL['Baseline_Predictions'] = baseline_predictions
DATA_PKL['Baseline_Actuals'] = baseline_actuals

if TUNE:
	print('CREATING STUDY')
	study = optimize_hyperparameters(
		train_dataloader,
		val_dataloader,
		model_path=f'{output_dir}/Tuning/OPTUNA_STUDY/OPTUNA_TUNING',
		log_dir = f'{output_dir}/Tuning/OPTUNA_STUDY/OPTUNA_LOGS',
		max_epochs=40,
		n_trials=25,
		gradient_clip_val_range=(0.01, 1.0),
		hidden_size_range=(30, 128),
		hidden_continuous_size_range=(30, 128),
		attention_head_size_range=(1, 4),
		learning_rate_range=(1e-4, 0.1),
		dropout_range=(0.1, 0.3),
		reduce_on_plateau_patience=4,
		use_learning_rate_finder=False, 
		verbose=2,
		optimizer='adam',
	)
	try:
		write_pickle('/mnt/locutus/remotesensing/r62/fire_danger/Results/Tuning/STUDY.pkl', study)
	except:
		print('PICKLING AN OPTUNA STUDY DOES NOT WORK')
else:

	study = read_pickle('/mnt/locutus/remotesensing/r62/fire_danger/Results/Tuning/STUDY.pkl')
	
best_hypers = study.best_trial.params
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  
logger = TensorBoardLogger('/mnt/locutus/remotesensing/r62/fire_danger/Results/Tuning/lightning_logs')

trainer = pl.Trainer(
	max_epochs=100,
	accelerator='cpu', # was gpu
	devices=1,
	enable_model_summary=True,
	gradient_clip_val=best_hypers['gradient_clip_val'],
	callbacks=[lr_logger, early_stop_callback],
	logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
	training,
	learning_rate=best_hypers['learning_rate'],
	hidden_size=best_hypers['hidden_size'],
	attention_head_size=best_hypers['attention_head_size'],
	dropout=best_hypers['dropout'], 
	hidden_continuous_size=best_hypers['hidden_continuous_size'],
	output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
	loss=QuantileLoss(),
	log_interval=10, 
	reduce_on_plateau_patience=5,
)

print('TRAINING MODEL')
trainer.fit(
	tft,
	train_dataloaders=train_dataloader,
	val_dataloaders=val_dataloader,
)

best_model_path = trainer.checkpoint_callback.best_model_path
print('BEST MODEL PATH:', best_model_path)
DATA_PKL['Best_Model_Path'] = best_model_path

best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
write_pickle(f'{output_dir}/Output', DATA_PKL)
