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
from sklearn.model_selection import GroupKFold, train_test_split, GroupShuffleSplit
from scipy.stats import pearsonr
import pickle
import shutil
from functools import partial
from typing import Callable

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# GLOBAL VARIABLES
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "config.yaml"
np.random.seed(42)


def train_model(
    name: str,
    phenotype: str,
    fixed: dict,
    search_space: dict,
    modelObj: Callable,
    data_path: str,
    hyp_settings: dict,
    model_id: str,
    n_splits: int = 10,
) -> None:
    """
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
    """
    print("STARTING TRAINING")
    # read data and prepare it
    data = pd.read_feather(data_path)
    X, Y, ringnrs = prep_data_before_train(data, phenotype)
    # create results dataframe, model performance of each fold is saved here
    results = pd.DataFrame(columns=["name", "phenotype", "fold", "corr"])
    # if modelObj is string, then we are using INLA
    is_INLA = isinstance(modelObj, str)
    # create cross validation folds
    kf = GroupKFold(n_splits=n_splits)
    for fold, (train_val_index,
               test_index) in enumerate(kf.split(X, groups=ringnrs)):
        print(f"Starting fold {fold}")
        # Split data into train_val and test
        X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
        Y_train_val, Y_test = Y.iloc[train_val_index], Y.iloc[test_index]
        ringnr_train_val = ringnrs.iloc[train_val_index]
        # if INLA, then we need to save indexes for the INLA script
        if is_INLA:
            # save indexes for INLA for each fold
            save_to_INLA(train_val_index, test_index, fold, ringnrs)
        else:
            # train model with hyperparameter optimization
            model = hyperopt_train(X_train_val, Y_train_val, ringnr_train_val,
                                   search_space, fixed, modelObj, hyp_settings)
            # save model for each fold
            save_CVfold(model, name, fold)
            # evaluate model on test set
            Y_preds = model.predict(X_test)
            corr = pearsonr(Y_test, Y_preds)[0]
            mse = np.mean((Y_test - Y_preds)**2)
            print(f"FOLD {fold} FINISHED\t corr: {corr}\t mse: {mse}")
            if fold == 0:
                results = pd.DataFrame(
                    {
                        "name": name,
                        "phenotype": phenotype,
                        "fold": fold,
                        "corr": corr,
                        "model_id": model_id
                    },
                    index=[0])
            else:
                results.loc[len(
                    results.index)] = [name, phenotype, fold, corr, model_id]
    # when all folds are done, save result
    save_run(name, results, is_INLA)


def train_between_pop(
    name: str,
    phenotype: str,
    fixed: dict,
    search_space: dict,
    modelObj: Callable,
    data_path: str,
    hyp_settings: dict,
    model_id: str,
) -> None:

    print("STARTING ACROSS POP TRAINING")
    data = pd.read_feather(data_path)
    X, Y, ringnrs = prep_data_before_train(data, phenotype)

    # TODO: Finish this
    outer_indexes = X.index[X.hatchisland.isin([22, 23, 24])]
    X_outer, X_inner = X.loc[outer_indexes], X.drop(outer_indexes,
                                                    errors="ignore")
    Y_outer, Y_inner = Y.loc[outer_indexes], Y.drop(outer_indexes,
                                                    errors="ignore")
    ringnr_outer, ringnr_inner = ringnrs.loc[outer_indexes], ringnrs.drop(
        outer_indexes, errors="ignore")
    # Train on outer, test on inner
    model_outer = hyperopt_train(X_outer, Y_outer, ringnr_outer, search_space,
                                 fixed, modelObj, hyp_settings)
    save_CVfold(model_outer, name, "outer")

    Y_preds = model_outer.predict(X_inner)
    corr_outer = pearsonr(Y_inner, Y_preds)[0]
    mse_outer = np.mean((Y_inner - Y_preds)**2)
    print(f"OUTER FINISHED\t corr: {corr_outer}\t mse: {mse_outer}")
    # Train on inner, test on outer
    model_inner = hyperopt_train(X_inner, Y_inner, ringnr_inner, search_space,
                                 fixed, modelObj, hyp_settings)
    save_CVfold(model_inner, name, "inner")
    Y_preds = model_inner.predict(X_outer)
    corr_inner = pearsonr(Y_outer, Y_preds)[0]
    mse_inner = np.mean((Y_outer - Y_preds)**2)
    print(f"INNER FINISHED\t corr: {corr_inner}\t mse: {mse_inner}")
    results = pd.DataFrame(
        {
            "name": [name, name],
            "phenotype": [phenotype, phenotype],
            "fold": ["outer", "inner"],
            "corr": [corr_outer, corr_inner],
            "model_id": [model_id, model_id],
        },
        index=[0, 1])
    save_run(name, results, False)


def objective(
    hyperparameters: dict,
    fixed: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    modelObj: Callable,
) -> dict:
    """
    objective function for hyperparameter optimization, trains model and returns correlation on validation set
    :param hyperparameters: hyperparameters to be optimized
    :param fixed: fixed hyperparameters
    :param X_train: training covariates
    :param Y_train: training phenotypes/targets
    :param X_val: validation covariates
    :param Y_val: validation phenotypes/targets
    :param modelObj: model object
    :return: dict containing negative correlation on validation set and a status object
    """
    merged_hyperparameters = {**hyperparameters, **fixed}
    model = modelObj(**merged_hyperparameters)
    model.fit(X_train, Y_train)
    Y_preds = model.predict(X_val)
    # corr = pearsonr(Y_val, Y_preds)[0]
    mse = np.mean((Y_preds - Y_val)**2)
    return {"loss": mse, "status": STATUS_OK}


def hyperopt_train(
    X_train_val: pd.DataFrame,
    Y_train_val: pd.Series,
    ringnr_train_val: pd.Series,
    search_space: dict,
    fixed: dict,
    modelObj: Callable,
    hyp_settings: dict,
) -> Callable:
    """
    hyperparameter optimization using hyperopt
    :param X_train_val: train and validation covariates
    :param Y_train_val: train and validation phenotypes/targets
    :param ringnr_train_val: ringnrs for the train and validation set
    :param search_space: dictionary containing the search space for hyperparameter optimization
    :param fixed: fixed hyperparameters
    :param modelObj: model object to be trained
    :param hyp_settings: settings for hyperopt
    :return: model object with optimized hyperparameters
    """
    trials = Trials()

    kf = GroupShuffleSplit(test_size=0.2, n_splits=2)
    split = kf.split(X_train_val, groups=ringnr_train_val)
    train_inds, val_inds = next(split)
    X_train, X_val = X_train_val.iloc[train_inds], X_train_val.iloc[val_inds]
    Y_train, Y_val = Y_train_val.iloc[train_inds], Y_train_val.iloc[val_inds]

    objective_partial = partial(
        objective,
        fixed=fixed,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        modelObj=modelObj,
    )
    best = fmin(
        fn=objective_partial,
        space=search_space,
        trials=trials,
        algo=tpe.suggest,
        **hyp_settings,
    )
    model = modelObj(**{**best, **fixed})
    model.fit(X_train_val, Y_train_val)
    return model


def save_CVfold(model: Callable, name: str, fold: int) -> None:
    """
    save model for each fold as pickle file
    :param model: trained model object to be saved
    :param name: name of the model
    :param fold: CV fold number
    :return: none
    """
    save_path = PROJECT_DIR / "models" / f"{name}"
    os.makedirs(save_path, exist_ok=True)
    with open(save_path / f"fold_{fold}.pkl", "wb") as file:
        pickle.dump(model, file)


def save_run(name: str, results: pd.DataFrame, is_INLA: bool = False) -> None:
    """
    save results of the run in the results dataframe, also copy config file to the model folder
    :param name: model name
    :param results: results dataframe
    :param is_INLA: if the model is INLA, then we don't want to save the results, as this is done in the INLA script
    :return: none
    """
    save_path = PROJECT_DIR / "models" / f"{name}"
    shutil.copyfile(CONFIG_PATH, save_path / "config.yaml")
    if not is_INLA:
        try:
            old_results = pd.read_pickle(PROJECT_DIR / "models" /
                                         "results.pkl")
            merged_results = pd.concat([old_results, results], axis=0)
            merged_results.to_pickle(PROJECT_DIR / "models" / "results.pkl")
        except FileNotFoundError:
            results.to_pickle(PROJECT_DIR / "models" / "results.pkl")


def save_to_INLA(train_val_index: np.ndarray, test_index: np.ndarray,
                 fold: int, ringnrs: pd.Series) -> None:
    """
    save ringnr indexes for INLA for a fold
    :param train_val_index: indexes for train_val set
    :param test_index: indexes for test set
    :param fold: cross validation fold number
    :param ringnrs: the ringnrs for each individual
    :return: none
    """
    ringnrs.iloc[train_val_index].to_frame().reset_index().to_feather(
        PROJECT_DIR / "data" / "interim" / f"ringnr_train_{fold}.feather")
    ringnrs.iloc[test_index].to_frame().reset_index().to_feather(
        PROJECT_DIR / "data" / "interim" / f"ringnr_test_{fold}.feather")


def main():
    with open(CONFIG_PATH, "r") as file:
        configs = yaml.safe_load(file)
    args = handle_yaml_before_train(**configs)
    train_across = args.pop("train_across")
    if train_across:
        train_between_pop(**args)
    else:
        train_model(**args)


if __name__ == "__main__":
    main()
