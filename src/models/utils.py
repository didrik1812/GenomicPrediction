from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pathlib import Path
from src.features import searchspaces
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_DICT = {
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
    "INLA": "INLA"
}

DATA_PATHs = {
    "bodymass two-step": PROJECT_DIR / "data" / "processed" / "massBV.feather",
    "bodymass one-step": PROJECT_DIR / "data" / "processed" / "massEG.feather",
    "tarsus two-step": PROJECT_DIR / "data" / "processed" / "tarsusBV.feather",
    "tarsus one-step": PROJECT_DIR / "data" / "processed" / "tarsusEG.feather",
}


def handle_yaml_before_train(name: str, phenotype: str, model: str, procedure: str, searchspace: str, fixed: dict, hyp_settings:dict)->dict:
    '''
    create a dictionary with all the information from config file needed for training
    :param name: name of the model
    :param phenotype: phenotype to be predicted
    :param model: chosen model
    :param procedure: two-step or one-step procedure
    :param searchspace: hyperparameter search space
    :param fixed: fixed hyperparameters
    :return: dict
    '''
    search_space = searchspaces.search_spaces[searchspace]
    modelObj = MODEL_DICT[model]
    data_path = DATA_PATHs[phenotype + " " + procedure]

    return {"name": name, "phenotype": phenotype, "fixed": fixed,
            "search_space": search_space, "modelObj": modelObj, "data_path": data_path, "hyp_settings":hyp_settings}


def prep_data_before_train(data: pd.DataFrame, phenotype: str)->tuple:
    '''
    prepare data for training, returns target vector and covariate-matrix and ringnrs for grouping
    :param data: all data from dataloader script
    :param phenotype: the phenotype to be predicted
    :return: X, Y, ringnrs, X contain covariates and Y contains the phenotype
    '''
    X = data.drop(columns=["ID", "mass", "tarsus",
                           "mean_pheno", "FID", "MAT", "PAT", "SEX", "PHENOTYPE"], errors="ignore")
    try:
        Y = data.loc[:, phenotype]
    except KeyError:
        Y = data.ID

    try:
        X.hatchyear = X.hatchyear.astype("int")
        X.island_current = X.island_current.astype("int")
    except AttributeError:
        pass
    ringnrs = data.ringnr
    return X, Y, ringnrs
