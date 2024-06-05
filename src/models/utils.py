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
from sklearn.linear_model import ElasticNet
import yaml
import numpy as np
from .CustomModels.LinearResidTree import LinearResidTree
from .CustomModels.HuberContamination import HuberContamination 
from .CustomModels.LinearLGBM import LinearLGBM

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
            "IID",
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
    Y = (Y - np.mean(Y))/np.std(Y)
    X = X.fillna(0)
    X.loc[:, snp_cols] = X.loc[:, snp_cols].astype("int")
    ringnrs = data.ringnr
    mean_pheno = data.mean_pheno
    return X, Y, ringnrs, mean_pheno



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
    fold: Union[int, str]
    X_train: pd.DataFrame = pd.DataFrame()
    y_train: pd.DataFrame = pd.DataFrame()
    X_val: pd.DataFrame  = pd.DataFrame()
    y_val: pd.DataFrame  = pd.DataFrame()
    ringnr_train: pd.DataFrame = pd.DataFrame()
    ringnr_val: pd.DataFrame = pd.DataFrame()
    mean_pheno_test: pd.DataFrame = pd.DataFrame()

    def __post_init__(self):
        kf = GroupShuffleSplit(n_splits=2)
        split = kf.split(self.X_train_val, self.y_train_val, groups=self.ringnr_train_val)
        train_inds, val_inds = next(split)
        self.X_train = self.X_train_val.iloc[train_inds]
        self.y_train = self.y_train_val.iloc[train_inds]
        self.X_val = self.X_train_val.iloc[val_inds]
        self.y_val = self.y_train_val.iloc[val_inds]
        self.ringnr_val = self.ringnr_train_val.iloc[val_inds]
        self.ringnr_train = self.ringnr_train_val.iloc[train_inds]

    def winsorize(self):
        from scipy.stats.mstats import winsorize
        self.y_train = winsorize(self.y_train , limits = [0.1, 0.1]).data

    def reduce_snps(self, num_snps:int = 500):
        snp_cols = [c for c in self.X_train.columns if c.startswith("SNP")]
        non_snp_cols = [c for c in self.X_train.columns if c not in snp_cols]
        # find correlation of snp and sort it in ascending order
        snp_rank = abs(self.X_train.loc[:, snp_cols].corrwith(self.y_train)).sort_values(ascending = False)
        # select subset of snps
        snp_subset = snp_rank[:num_snps].index.to_list()
        self.X_train = self.X_train.loc[:, snp_subset + non_snp_cols]
        self.X_val = self.X_val.loc[:, snp_subset + non_snp_cols]
        self.X_test = self.X_test.loc[:, snp_subset + non_snp_cols]

    def add_interaction(self, num_snps:int = 500):
        self.reduce_snps(num_snps)
        snp_cols = [c for c in self.X_train.columns if c.startswith("SNP")]
        for snp1 in snp_cols:
            for snp2 in snp_cols:
                self.X_train[f"{snp1}_{snp2}"] = self.X_train.loc[:, snp1].to_array() * self.X_train.loc[:, snp2].to_array()  
                self.X_val[f"{snp1}_{snp2}"] = self.X_val.loc[:, snp1].to_array() * self.X_val.loc[:, snp2].to_array()  
                self.X_test[f"{snp1}_{snp2}"] = self.X_test.loc[:, snp1].to_array() * self.X_test.loc[:, snp2].to_array()  
 
        



@dataclass
class ModelConfig:
    """
    Dataclass for holding the model configurations
    """

    project_path: Path
    yaml_path: Path

    data_path = Path() 
    name = "" 
    model_id= "" 
    phenotype= "" 
    model: Union[XGBRegressor, CatBoostRegressor, LGBMRegressor, str] = XGBRegressor 
    procedure = "" 
    searchspace = {} 
    fixed_params = {} 
    hyp_settings = {} 
    train_across= False 
    search_space= {} 
    train_across_islands = False

    def __post_init__(self):
        self.model_dict = {
            "xgboost": XGBRegressor,
            "lightgbm": LGBMRegressor,
            "catboost": CatBoostRegressor,
            "linearresidtree": LinearResidTree,
            "hubercontamination": HuberContamination,
            "linearlgbm": LinearLGBM,
            "INLA": "INLA",
            "mean": "mean",
            "elasticnet": ElasticNet,
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
            "tarsus delta": self.project_path
            / "data"
            / "processed"
            / "tarsusEG.feather",
            "bodymass delta": self.project_path
            / "data"
            / "processed"
            / "massEG.feather",
            "bodymass_70k two-step": self.project_path
            / "data"
            / "processed"
            / "massBV_70k.feather",
            "tarsus_70k two-step": self.project_path
            / "data"
            / "processed"
            / "tarsusBV_70k.feather",


        }
        self.handle_yaml()

    def handle_yaml(self):
        with open(self.yaml_path, "r") as f:
            config = yaml.safe_load(f)

        self.data_path = self.data_paths[
            config["phenotype"] + config.get("data", "") + " " + config["procedure"]
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
        self.train_across_islands = config["train_across_islands"]


