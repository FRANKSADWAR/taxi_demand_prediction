import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.pardir)))
import lightgbm as LGB

import pandas as pd
import numpy  as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from paths import TRANSFORMED_DATA_DIR
from data_split import train_test_split

import numpy as np
import optuna
from optuna.visualization import plot_intermediate_values
from model import get_pipeline
from sklearn.model_selection import KFold, TimeSeriesSplit

df = pd.read_parquet(TRANSFORMED_DATA_DIR / 'tarbular_data.parquet')


X_train, y_train, X_test, y_test = train_test_split(
    df,
    cutoff_date=datetime(2022,6,1,0,0,0),
    target_column_name='target_rides_next_hour'
)
print(f'{X_train.shape=}')
print(f'{y_train.shape=}')
print(f'{X_test.shape=}')
print(f'{y_test.shape=}')

def objective(trial: optuna.trial.Trial) -> float:
    """
    Given a set of hyper-parameters, it trains a model and computes an average
    validation error based on a TimeSeriesSplit
    """
    ## Pick the hyper-parameters
    hyperparams = {
        "metric":"mae",
        "verbose": -1,
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 100)
    }

    tss = TimeSeriesSplit(n_splits=4)
    scores = []
    for train_index, val_index in tss.split(X_train):

        ## Split data for training and validation
        X_train_, X_val_ = X_train.iloc[train_index, :], X_train.iloc[val_index,:]
        y_train_, y_val_ = y_train.iloc[train_index], X_train.iloc[val_index]
        
        ## Train the model
        pipeline = get_pipeline(**hyperparams)
        
        pipeline.fit(X_train_, y_train_)

        print(len(X_train_.columns))

        ## Evaluate the model
        y_pred = pipeline.predict(X_val_)
        mae = mean_absolute_error(y_val_, y_pred)

        scores.append(mae)

    ## Return the mean mae
    print(scores)
    return np.array(scores).mean()
    

## Now we create a study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)