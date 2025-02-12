#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import keras_tuner as kt

class ResidualConnection(tf.keras.layers.Layer):
    def call(self, inputs):
        inputs = tf.keras.layers.Lambda(lambda x: x[:, -1, 2])(inputs)
        return tf.expand_dims(inputs, axis=-1)

class Transformer(kt.HyperModel):

    def __init__(self, 
                 input_shape,
                 # batch_size,
                 optimizer,
                 loss,
                 sub_batch_size,
                 static_vars,
                 metrics = None,
                 momentum = None,
                 strategy = None):
        self.input_shape = input_shape
        # self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.sub_batch_size = sub_batch_size
        self.static_vars = static_vars
        self.metrics = metrics
        self.momentum = momentum
        self.strategy = strategy
        
        # Not sure if this is working:
        self.number_of_gpus = len(tf.config.list_physical_devices('GPU'))
        
    def find_factors(self, number):
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
        
    def convert_loss(self):

        # Regression:
        if self.loss.upper() == 'MSLE':
            loss_fn = tf.keras.losses.MeanSquaredLogarithmicError(name="msle")
        elif self.loss.upper() == 'MSE':
            loss_fn = tf.keras.losses.MeanSquaredError(name='mse')
        elif self.loss.upper() == 'HUBER':
            loss_fn = tf.keras.losses.Huber(delta=1.0, name='huber')
        elif (self.loss.upper() == 'MAE') or (self.loss.upper() == 'LOSS'):
            loss_fn = tf.keras.losses.MeanAbsoluteError(name='mae')

        # Classification
        elif self.loss.upper() == 'BCE':
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, 
                                                         name = 'binary_cross_entropy')
        elif self.loss.upper() == 'SCCE':
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, 
                                                         name = 'sparse_categorical_cross_entropy')
        elif self.loss.upper() == 'CCE':
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, 
                                                         name = 'categorical_cross_entropy')
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
        return loss_fn

    def produce_direction(self):

        min_metrics = set(['MSLE', 'RMSE', 'MAE', 'MSE', 'LOG_COSH_ERROR', 'COS_SIM'])
        max_metrics = set(['AUC', 'ACCURACY', 'CATEGORICAL_ACCURACY', 
                           'SPARSE_CATEGORICAL_ACCURACY', 'PRECISION', 
                           'RECALL', 'F1_SCORE'])
        metric_names = set([metric.name.upper() for metric in self.metrics])

        if metric_names.issubset(min_metrics):
            direction = 'min'
        elif metric_names.issubset(max_metrics):
            direction = 'max'
        else:
            direction = None
            print('CHECK YOUR METRICS. COULD BE SPELLED DIFFERENTLY; UNAVAILABLE; MIXED BETWEEN MIN AND MAX')

        return direction

    def random_select(self, X, y):

        n = X.shape[0]
        if n > self.sub_batch_size:
            indices = np.random.choice(n, size=self.sub_batch_size, replace=False)
            selected_X = X[indices]
            selected_y = y[indices]
            selected_static = self.static_vars[indices]
        else:
            selected_X = X
            selected_y = y
            selected_static = self.static_vars

        return selected_X, selected_y, selected_static


    def generator(self, files, batch_size):

        mask3 = ~np.any(np.isnan(self.static_vars), axis=(1, 2))

        while True:
            for idx, file in enumerate(files):
                if idx == int(len(files) - 1):
                    continue

                mmap_arr_X = np.load(file) # mmap_mode = 'r'
                mmap_arr_X = mmap_arr_X.astype('float32')
                mask1 = ~np.any(np.isnan(mmap_arr_X), axis=(1, 2))

                mmap_arr_y = np.load(files[idx+1]) # mmap_mode = 'r'
                mmap_arr_y = mmap_arr_y[:, -1, 2].astype('float32')
                mask2 = ~np.isnan(mmap_arr_y)

                mask = mask1*mask2*mask3

                mmap_arr_X = mmap_arr_X[mask]         
                mmap_arr_y = mmap_arr_y[mask]
                static_vars_masked = self.static_vars[mask]

                mmap_arr_X, mmap_arr_y, static_vars_masked = self.random_select(mmap_arr_X, 
                                                                    mmap_arr_y)

                mmap_arr_X = np.concatenate([mmap_arr_X, static_vars_masked], axis = 2).astype('float32')
                splits = np.ceil(mmap_arr_X.shape[0]/batch_size)
                split_X = np.array_split(mmap_arr_X, splits, axis = 0)
                split_y = np.array_split(mmap_arr_y, splits, axis = 0)

                for X, y in zip(split_X, split_y):
                    # print(X.shape, y.shape)
                    yield X, y
    
    def transformer_encoder(self, inputs, attention_head_size, number_of_heads, feed_forward_dimensions, transformer_stack_dropout):
        # Attention and Normalization
        x = tf.keras.layers.MultiHeadAttention(key_dim=attention_head_size, num_heads=number_of_heads, dropout=transformer_stack_dropout)(inputs, inputs)
        x = tf.keras.layers.Dropout(transformer_stack_dropout)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.Conv1D(filters=feed_forward_dimensions, kernel_size=1, activation="relu")(res)
        x = tf.keras.layers.Dropout(transformer_stack_dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model_body(self, hp, parameters=None):
        
        param_keys = ['batch_size', 'num_encoder_blocks', 
                      'mlp_layers', 'mlp_units', 'mlp_activation_function',
                       'mlp_dropout', 'transformer_stack_dropout', 'attention_head_size', 'number_of_heads',
                       'feed_forward_dimensions', 'learning_rate']
        
        if not parameters:
            
            # attention heads are generally divisible by 
            possible_number_of_heads = self.find_factors(self.input_shape[-1])
            attention_head_scale_parameter = 8
            
            if len(possible_number_of_heads) == 2:
                # meaning its a prime THIS IS JUST A PLACE HOLDER FOR PRIMES!!! PUT SOMETHING ELSE HERE DEPENDING ON PROBLEM!!
                possible_number_of_heads = hp.Int("number_of_heads", 3, 5)
            else:
                possible_number_of_heads = hp.Choice("number_of_heads", possible_number_of_heads[1:-1])
            
            if self.number_of_gpus > 0:
                batch_size = hp.Int("batch_size", min_value=100*self.number_of_gpus, 
                                    max_value=500*self.number_of_gpus, 
                                    step=100*self.number_of_gpus)
            else:
                batch_size = hp.Int("batch_size", min_value=96, max_value=352, step=64) 
                
            parameters = {'num_encoder_blocks' : hp.Int("encoder_blocks", 2, 4),
                          'mlp_layers' : hp.Int("mlp_layers", 1, 2),
                          'mlp_units' : hp.Int("mlp_units", 16, 256, 16),
                          'mlp_activation_function' : hp.Choice("mlp_activation_function", ["leaky_relu", 'relu', 'tanh']),
                          'mlp_dropout' : hp.Float('mlp_dropout', min_value=0, max_value=0.5, step = 0.05),
                          'transformer_stack_dropout' : hp.Float('transformer_stack_dropout', min_value=0, max_value=0.5, step = 0.05),
                          'attention_head_size' : possible_number_of_heads*attention_head_scale_parameter, 
                          #hp.Int(f'head_size', min_value=16, max_value=256, step=16),
                          'number_of_heads' : possible_number_of_heads,
                          'feed_forward_dimensions' : hp.Int("feed_forward_dimensions", 2*self.input_shape[-1], 4*self.input_shape[-1], step = 16),
                          'learning_rate' : hp.Float('learning_rate',  min_value=1e-5, max_value=0.1, step = 10, sampling='log')
                         }
        elif (isinstance(parameters, dict)) and (all(key in parameters for key in param_keys)):
            pass

        else:
            raise ValueError(f'"parameters" must be a dictionary containing all of the following keys: \n\n{param_keys}')
        
        inputs = tf.keras.Input(shape=self.input_shape) # batch_size = parameters['batch_size']
        
        # making the fire danger project a residual connection
        current_scores = inputs
        current_scores = ResidualConnection()(current_scores)

        x = inputs
        for _ in range(parameters['num_encoder_blocks']):
            x = self.transformer_encoder(x, parameters['attention_head_size'], parameters['number_of_heads'], 
                                         parameters['feed_forward_dimensions'], parameters['transformer_stack_dropout'])

        # Adding LSTM layer for maintaining memory
        # if hp.Boolean("LSTM_Layer"):
        #     x = tf.keras.layers.LSTM(units=32, return_sequences=True)(x)
        
        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for _ in range(parameters['mlp_layers']):
            x = tf.keras.layers.Dense(parameters['mlp_units'],
                                   activation=parameters['mlp_activation_function'])(x)
            x = tf.keras.layers.Dropout(parameters['mlp_dropout'])(x)

        # Change this to fit the problem at hand
        x = x + current_scores
        
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        
        model = tf.keras.Model(inputs, outputs)

        if self.optimizer.upper() == 'ADAM':
            # Notice momentum is not being used if the compiler is Adam
            opt=tf.keras.optimizers.Adam(learning_rate = parameters['learning_rate'])		

        elif self.optimizer.upper() == 'SGD':
            opt = tf.keras.optimizers.SGD(learning_rate = parameters['learning_rate'], 
                                          momentum = self.momentum)

        if self.metrics == None:
            model.compile(loss = self.convert_loss(), optimizer = opt)
        else:
            model.compile(loss = self.convert_loss(), optimizer = opt, metrics=self.metrics)

        return model

    def build(self, hp, parameters=None):
        self.parameters = parameters
        if self.strategy:
            with self.strategy.scope():
                if self.parameters:
                    model = self.build_model_body(hp=None, parameters = self.parameters)
                else:
                    model = self.build_model_body(hp)
        else:
            if self.parameters:
                model = self.build_model_body(hp=None, parameters = self.parameters)
            else:
                model = self.build_model_body(hp)
        return model

    def fit(self, hp, model, training_data, validation_data, **kwargs): # *args
        # where training_data and validation_data are actually 
        # the absolute paths to the binary files containing data
        
        if self.parameters:
            batch_size = self.parameters['batch_size']*self.number_of_gpus
        else:
            batch_size = hp.get('batch_size')
        
        train_n = int(len(training_data)*self.sub_batch_size)
        val_n = int(len(validation_data)*self.sub_batch_size)
        
        return model.fit(
            self.generator(training_data, batch_size), # this serves as X, y for training
            validation_data = self.generator(validation_data, batch_size),
            steps_per_epoch = int(np.ceil(train_n/(batch_size))),
            validation_steps = int(np.ceil(val_n/(batch_size))),
            # Tune whether to shuffle the data in each epoch.
            # shuffle = hp.Boolean("shuffle"),
            batch_size = batch_size,
            # *args,
            **kwargs)

