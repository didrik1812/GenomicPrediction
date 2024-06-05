'''
get_shap.py
This script is used to obtain SHAP values from a tree-based XGBoost Model
Modify the "mod_path" variable to the loccation of your models to obtain SHAP values
The SHAP values is stored in the file "shap.feather" under the same directory as your model
Run it by "python -m src.models.get_shap"
'''

import lightgbm as lgbm
import xgboost as xgb
from pathlib import Path
import shap
import numpy as np
from .utils import (
    ModelConfig,
    prep_data_before_train
)
from .modules import (
        ModelCV
)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

mod_path = Path("/work/didrikls/GenomicPrediction/models/xgboost2MassStd_70kExtBV")
np.random.seed(42)
mods = []
# we extract the config file of the model to obtain the correct datasets
yaml_path = mod_path / "config.yaml"
modelSettings = ModelConfig(mod_path.parents[1], yaml_path)
data_path = modelSettings.data_path
mcv = ModelCV.ModelCV(data_path, modelSettings)

# Extract the models from a cross-validation
for model_path in sorted(mod_path.glob("*.json")):
    logging.info(model_path)
    mod = xgb.XGBRegressor()
    mod.load_model(model_path)
    mods.append(mod)


logging.info(f"{len(mods)} models of {mod_path.stem} loaded")

# Extract data used for the models
data = pd.read_feather(data_path)


X, y, ringnrs, mean_pheno = prep_data_before_train(
    data=data, phenotype=modelSettings.phenotype
)
if modelSettings.procedure == "two-step":
    X = X.drop(columns = "hatchisland")


logging.info("Starting SHAP")
shap_values_list = []
# We only do SHAP on the test set of the model. 
for i, (train_val_index, test_index, fold) in enumerate(mcv.splitter(X, ringnrs)):

    X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
    y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
    ringnr_train_val, ringnr_test = ringnrs.iloc[train_val_index], ringnrs.iloc[test_index]
    # We use the path-dependent algorithm in Lundeberg et.al 2020, therefore no background dataset is needed
    explainer = shap.TreeExplainer(mods[i])
    shap_values = explainer.shap_values(X_test)


    shap_values_list.append(shap_values)

    logging.info(f"Fold {i} finished")

logging.info("SHAP value computation completed")

# Concatenate SHAP values from all folds
all_shap_values = np.concatenate(shap_values_list, axis=0)

# Compute the mean of the absolute SHAP values along the zero axis
mean_shap = np.abs(all_shap_values).mean(axis=0)

# Ensure the mean_shap is a 1D array with shape (n_features,)
if mean_shap.ndim == 2:
    mean_shap = mean_shap.flatten()

# Create a DataFrame from the mean SHAP values and save it 
shap_df = pd.DataFrame(mean_shap.reshape(1, -1), columns=X.columns)

shap_df.to_feather(mod_path / "shap2.feather")
