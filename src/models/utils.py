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
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]


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
    """
    Class for holding the data for a single fold
    Splits the training data into train and validation set

    param X_train_val: pd.DataFrame
    param y_train_val: pd.DataFrame
    param X_test: pd.DataFrame
    param y_test: pd.DataFrame
    param ringnr_train_val: pd.DataFrame
    param ringnr_test: pd.DataFrame
    param fold: Union[int, str] = None
    """

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
    """
    Dataclass for holding the model configurations
    """

    project_path: Path
    yaml_path: Path

    def __post_init__(self):
        self.model_dict = {
            "xgboost": XGBRegressor,
            "lightgbm": LGBMRegressor,
            "catboost": CatBoostRegressor,
            "INLA": "INLA",
        }

        self.data_paths = {
            "bodymass two-step": self.project_path
            / "data"
            / "processed"
            / "massBV.feather",
            "bodymass one-step": self.project_path
            / "data"
            / "processed"
            / "massEG.feather",
            "tarsus two-step": self.project_path
            / "data"
            / "processed"
            / "tarsusBV.feather",
            "tarsus one-step": self.project_path
            / "data"
            / "processed"
            / "tarsusEG.feather",
        }
        self.handle_yaml()

    def handle_yaml(self):
        with open(self.yaml_path, "r") as f:
            config = yaml.safe_load(f)

        self.data_path = self.data_paths[
            config["phenotype"] + " " + config["procedure"]
        ]
        self.name = config["name"]
        self.model_id = config["model"]
        self.phenotype = config["phenotype"]
        self.model = self.model_dict[config["model"]]
        self.procedure = config["procedure"]
        self.searchspace = searchspaces.search_spaces[config["searchspace"]]
        self.fixed_params = config["fixed"]
        self.hyp_settings = config["hyp_settings"]
        self.train_across = config["train_across"]
        self.search_space = searchspaces.search_spaces[config["searchspace"]]
