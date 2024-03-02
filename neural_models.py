import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import pandas as pd
import numpy as np

from skopt import gp_minimize

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, r2_score

class NeuralNetworkModel:
    def __init__(self, input_dim, output_dim=1, task_type='price',
                 learning_rate=0.001, num_units=64, dropout_rate=0.3, num_layers=1,
                 hidden_activation='relu', output_activation='linear'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(self.input_dim,)))

        # Architecture based on task type
        if self.task_type == 'trim':
            model.add(layers.Dense(self.num_units, activation=self.hidden_activation))
            model.add(layers.Dropout(self.dropout_rate))
            for _ in range(self.num_layers):
                model.add(layers.Dense(self.num_units, activation=self.hidden_activation))
            model.add(layers.Dense(self.output_dim, activation=self.output_activation))  # Assuming 9 trims

            # Compile with categorical crossentropy and auc for trim classification
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                          loss='categorical_crossentropy', metrics=['AUC'])
        elif self.task_type == 'price':
            model.add(layers.Dense(self.num_units, activation=self.hidden_activation))
            model.add(layers.Dropout(self.dropout_rate))
            for _ in range(self.num_layers):
                model.add(layers.Dense(self.num_units, activation=self.hidden_activation))
            model.add(layers.Dense(self.output_dim, activation=self.output_activation))  # Linear activation for regression

            # Compile with mean squared error for price regression
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                          loss='mean_squared_error', metrics=[r2_metric])
        else:
            raise ValueError("Invalid task type. Supported types: 'trim' or 'price'.")

        return model

    def train(self, X_train, y_train, batch_size=32, epochs=10, validation_split=0.2):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def predict_one_trim(self, X):
        return np.argmax(self.model.predict(X), axis=-1)


def objective_function(params, X, y, task_type):
    # Unpack hyperparameters
    learning_rate, num_units, dropout_rate, num_layers, batch_size, hidden_activation, output_activation = params
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store scores for each fold
    scores = []

    for train_index, val_index in kf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        output_dim = len(np.unique(y))
        
        if task_type == 'trim':
            y_train_fold_onehot = to_categorical(y_train_fold, num_classes=output_dim)
        else:
            y_train_fold_onehot = y_train_fold  # For 'price' task, no one-hot encoding needed

        model = NeuralNetworkModel(input_dim=X_train_fold.shape[1], output_dim=output_dim,
                                   learning_rate=learning_rate, num_units=num_units,
                                   dropout_rate=dropout_rate, num_layers=num_layers,
                                   hidden_activation=hidden_activation,
                                   output_activation=output_activation,
                                   task_type=task_type)

        model.train(X_train_fold, y_train_fold_onehot, batch_size=int(batch_size))

        if task_type == 'trim':
            y_pred = model.predict(X_val_fold)
            y_pred_one_trim = model.predict_one_trim(X_val_fold)
             # Add one-hot encoding for y_val_fold as well
            y_val_fold_onehot = to_categorical(y_val_fold, num_classes=output_dim).reshape(-1, output_dim)  # Reshape to 2D
            y_pred = y_pred.reshape(-1, output_dim)  # Reshape to 2D
            
            score = roc_auc_score(y_val_fold_onehot, y_pred, multi_class='ovr')
        elif task_type == 'price':
            y_pred = model.predict(X_val_fold)
            score = r2_score(y_val_fold, y_pred)
        else:
            raise ValueError("Invalid task type. Supported types: 'trim' or 'price'.")

        scores.append(score)

    return -np.mean(scores)  # Negate due to optuna minimization and desire for max roc_auc

def r2_metric(y_true, y_pred):
    mean_actual = tf.reduce_mean(y_true)
    
    # Calculate SSR and SST
    ssr = tf.reduce_sum(tf.square(y_true - y_pred))
    sst = tf.reduce_sum(tf.square(y_true - mean_actual))
    
    # Calculate R^2
    r2 = 1 - (ssr / sst)
    
    return r2