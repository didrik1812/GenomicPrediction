import os

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import logging
import yaml
from .utils import handle_yaml_before_train, prep_data_before_train
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import GroupKFold, train_test_split
from scipy.stats import pearsonr
import pickle
import shutil
from functools import partial
from typing import Callable

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# GLOBAL VARIABLES
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "config.yaml"
np.random.seed(42)



def train_model(name: str, phenotype: str, fixed: dict, search_space: dict,
                modelObj: Callable, data_path: str, hyp_settings: dict, n_splits: int = 10) -> None:
    '''
    train an arbitrary model on the data, using the search space and the model object
    results are saved in the GenomicPrediction/models folder
    also stores indexes for INLA if that is choosen the model
    :param name: name of the model
    :param phenotype: phenotype to be predicted
    :param fixed: fixed hyperparameters
    :param search_space: search space for hyperparameter optimization
    :param modelObj: model object
    :param data_path: path to data
    :param n_splits: number of CV folds
    :return: none
    '''
    print('STARTING TRAINING')
    # read data and prepare it
    data = pd.read_feather(data_path)
    X, Y, ringnrs = prep_data_before_train(data, phenotype)
    # create results dataframe, model performance of each fold is saved here
    results = pd.DataFrame(columns=["name", "phenotype", "fold", "corr"])
    # if modelObj is string, then we are using INLA
    is_INLA = isinstance(modelObj, str)
    # create cross validation folds
    kf = GroupKFold(n_splits=n_splits)
    for fold, (train_val_index, test_index) in enumerate(kf.split(X, groups=ringnrs)):
        print(f"Starting fold {fold}")
        # Split data into train_val and test
        X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
        Y_train_val, Y_test = Y.iloc[train_val_index], Y.iloc[test_index]
        # if INLA, then we need to save indexes for the INLA script
        if is_INLA:
            # save indexes for INLA for each fold
            save_to_INLA(train_val_index, test_index, fold, ringnrs)
        else:
            # train model with hyperparameter optimization
            model = hyperopt_train(X_train_val, Y_train_val, search_space, fixed, modelObj, hyp_settings)
            # save model for each fold
            save_CVfold(model, name, fold)
            # evaluate model on test set
            Y_preds = model.predict(X_test)
            corr = pearsonr(Y_test, Y_preds)[0]
            mse = np.mean((Y_test - Y_preds) ** 2)
            print(f"FOLD {fold} FINISHED\t corr: {corr}\t mse: {mse}")
            results = results.append({"name": name, "phenotype": phenotype, "fold": fold, "corr": corr},
                                     ignore_index=True)
    # when all folds are done, save result
    save_run(name, results, is_INLA)


def objective(hyperparameters: dict,fixed:dict ,X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
              modelObj: Callable) -> dict:
    '''
    objective function for hyperparameter optimization, trains model and returns correlation on validation set
    :param hyperparameters: hyperparameters to be optimized
    :param fixed: fixed hyperparameters
    :param X_train: training covariates
    :param Y_train: training phenotypes/targets
    :param X_val: validation covariates
    :param Y_val: validation phenotypes/targets
    :param modelObj: model object
    :return: dict containing negative correlation on validation set and a status object
    '''
    merged_hyperparameters = {**hyperparameters, **fixed}
    model = modelObj(**merged_hyperparameters)
    model.fit(X_train, Y_train)
    Y_preds = model.predict(X_val)
    corr = pearsonr(Y_val, Y_preds)[0]
    return {'loss': -corr, 'status': STATUS_OK}


def hyperopt_train(X_train_val: np.ndarray, Y_train_val: np.ndarray, search_space: dict, fixed: dict,
                   modelObj: Callable, hyp_settings: dict) -> Callable:
    '''
    hyperparameter optimization using hyperopt
    :param X_train_val: train and validation covariates
    :param Y_train_val: train and validation phenotypes/targets
    :param search_space: dictionary containing the search space for hyperparameter optimization
    :param fixed: fixed hyperparameters
    :param modelObj: model object to be trained
    :param hyp_settings: settings for hyperopt
    :return: model object with optimized hyperparameters
    '''
    trials = Trials()
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state=42)
    objective_partial = partial(objective,fixed = fixed, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                                modelObj=modelObj)
    best = fmin(fn=objective_partial, space=search_space, trials=trials, algo=tpe.suggest, **hyp_settings)
    model = modelObj(**{**best, **fixed})
    model.fit(X_train_val, Y_train_val)
    return model


def save_CVfold(model: Callable, name: str, fold: int) -> None:
    '''
    save model for each fold as pickle file
    :param model: trained model object to be saved
    :param name: name of the model
    :param fold: CV fold number
    :return: none
    '''
    save_path = PROJECT_DIR / "models" / f"{name}"
    os.makedirs(save_path, exist_ok=True)
    with open(save_path / f"fold_{fold}.pkl", "wb") as file:
        pickle.dump(model, file)


def save_run(name: str, results: pd.DataFrame, is_INLA: bool = False) -> None:
    '''
    save results of the run in the results dataframe, also copy config file to the model folder
    :param name: model name
    :param results: results dataframe
    :param is_INLA: if the model is INLA, then we don't want to save the results, as this is done in the INLA script
    :return: none
    '''
    save_path = PROJECT_DIR / "models" / f"{name}"
    shutil.copyfile(CONFIG_PATH, save_path / "config.yaml")
    if not is_INLA:
        try:
            old_results = pd.read_pickle(PROJECT_DIR / "models" / "results.feather")
            merged_results = pd.concat([old_results, results], axis=0)
            merged_results.to_pickle(PROJECT_DIR / "models" / "results.feather")
        except FileNotFoundError:
            results.to_pickle(PROJECT_DIR / "models" / "results.feather")


def save_to_INLA(train_val_index: np.ndarray, test_index: np.ndarray, fold: int, ringnrs: pd.Series) -> None:
    '''
    save ringnr indexes for INLA for a fold
    :param train_val_index: indexes for train_val set
    :param test_index: indexes for test set
    :param fold: cross validation fold number
    :param ringnrs: the ringnrs for each individual
    :return: none
    '''
    ringnrs.iloc[train_val_index].to_frame().reset_index().to_feather(
        PROJECT_DIR / "data" / "interim" / f"ringnr_train_{fold}.feather")
    ringnrs.iloc[test_index].to_frame().reset_index().to_feather(
        PROJECT_DIR / "data" / "interim" / f"ringnr_test_{fold}.feather")


def main():
    with open(CONFIG_PATH, 'r') as file:
        configs = yaml.safe_load(file)
    args = handle_yaml_before_train(**configs)
    train_model(**args)


if __name__ == "__main__":
    main()
