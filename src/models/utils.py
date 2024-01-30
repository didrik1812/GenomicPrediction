"""
Path: src/models/utils.py
This file contains helper functions used for training and handling config files
"""
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pathlib import Path
from ..features import searchspaces
import pandas as pd
import os
from typing import Union
from dataclasses import dataclass
from sklearn.model_selection import GroupShuffleSplit

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_DICT = {
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
    "INLA": "INLA",
}

DATA_PATHs = {
    "bodymass two-step": PROJECT_DIR / "data" / "processed" / "massBV.feather",
    "bodymass one-step": PROJECT_DIR / "data" / "processed" / "massEG.feather",
    "tarsus two-step": PROJECT_DIR / "data" / "processed" / "tarsusBV.feather",
    "tarsus one-step": PROJECT_DIR / "data" / "processed" / "tarsusEG.feather",
}


def handle_yaml_before_train(
    name: str,
    phenotype: str,
    model: str,
    procedure: str,
    searchspace: str,
    fixed: dict,
    hyp_settings: dict,
    train_across: bool,
) -> dict:
    """
    create a dictionary with all the information from config file needed for training
    :param name: name of the model
    :param phenotype: phenotype to be predicted
    :param model: chosen model
    :param procedure: two-step or one-step procedure
    :param searchspace: hyperparameter search space
    :param fixed: fixed hyperparameters
    :return: dict
    """
    search_space = searchspaces.search_spaces[searchspace]
    modelObj = MODEL_DICT[model]
    data_path = DATA_PATHs[phenotype + " " + procedure]

    return {
        "name": name,
        "phenotype": phenotype,
        "fixed": fixed,
        "search_space": search_space,
        "modelObj": modelObj,
        "data_path": data_path,
        "hyp_settings": hyp_settings,
        "train_across": train_across,
        "model_id": model,
    }


def prep_data_before_train(data: pd.DataFrame, phenotype: str) -> tuple:
    """
    prepare data for training, returns target vector and covariate-matrix and ringnrs for grouping
    :param data: all data from dataloader script
    :param phenotype: the phenotype to be predicted
    :return: X, Y, ringnrs, X contain covariates and Y contains the phenotype
    """
    X = data.drop(
        columns=[
            "ID",
            "mass",
            "tarsus",
            "ringnr",
            "mean_pheno",
            "FID",
            "MAT",
            "PAT",
            "SEX",
            "PHENOTYPE",
        ],
        errors="ignore",
    )
    snp_cols = [c for c in X.columns if c.startswith("SNP")]
    try:
        Y = data.loc[:, phenotype]
    except KeyError:
        try:
            Y = data.ID
        except AttributeError:
            Y = data.mass

    try:
        X.hatchyear = X.hatchyear.astype("int")
        X.island_current = X.island_current.astype("int")
    except AttributeError:
        pass
    X = X.fillna(0)
    X.loc[:, snp_cols] = X.loc[:, snp_cols].astype("int")
    ringnrs = data.ringnr
    return X, Y, ringnrs


def get_current_model_names() -> tuple:
    """
    get all model names from the models folder
    :return: tuple(BVnames[list], EGnames[list], AcrossPopNames[list])
    """
    names = os.listdir(PROJECT_DIR / "models")
    BVnames = [name for name in names if name[-2:] == "BV"]
    EGnames = [name for name in names if name[-2:] == "EG"]
    AcrossPopNames = [name for name in names if name[-3:] == "EGA"]
    return BVnames, EGnames, AcrossPopNames


@dataclass
class Dataset:
    X_train_val: pd.DataFrame
    y_train_val: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    ringnr_train_val: pd.DataFrame
    ringnr_test: pd.DataFrame
    fold: Union[int, str] = None

    def __post_init__(self):
        kf = GroupShuffleSplit(n_splits=2)
        split = kf.split(self.X_train, self.y_train, groups=self.ringnr_train_val)
        train_inds, val_inds = next(split)
        self.X_train = self.X_train_val.iloc[train_inds]
        self.y_train = self.y_train_val.iloc[train_inds]
        self.X_val = self.X_train_val.iloc[val_inds]
        self.y_val = self.y_train_val.iloc[val_inds]


@dataclass
class ModelConfig:
    model: callable
    search_space: dict
    fixed_params: dict
    train_params: dict
    name: str
    model_id: str
    phenotype: str
