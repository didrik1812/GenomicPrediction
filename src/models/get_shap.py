import lightgbm as lgbm
import xgboost as xgb
from pathlib import Path
import pickle
import shap
import pltreeshap
import numpy as np
from .utils import (
    ModelConfig,
    prep_data_before_train
)
from .modules import (
        ModelCV
)
import pandas as pd
mod_path = Path("/work/didrikls/GenomicPrediction/models/linearlgbmTarsus_70kBV")
np.random.seed(42)
mods = []

yaml_path = mod_path / "config.yaml"
modelSettings = ModelConfig(mod_path.parents[1], yaml_path)
data_path = modelSettings.data_path
mcv = ModelCV.ModelCV(data_path, modelSettings)
# for model_path in mod_path.glob("*.json"):
#     print(model_path)
#     mod = xgb.XGBRegressor()
#     mod.load_model(model_path)
#     mods.append(mod)

for model_path in sorted(mod_path.glob("*.pkl")):
    print(model_path)
    with open(model_path, "rb") as mod:
        mods.append(pickle.load(mod))

print(f"{len(mods)} models loaded")


data = pd.read_feather(data_path)


X, y, ringnrs = prep_data_before_train(
    data=data, phenotype=modelSettings.phenotype
)
if modelSettings.procedure == "two-step":
    X = X.drop(columns = "hatchisland")

shap_values = np.zeros(shape = (len(mods), len(y), len(X.columns)))
print("starting shap")
# for i in range(len(mods)):
#     # explainer = shap.Explainer(mods[i])
#     explainer = shap.TreeExplainer(mods[i])
#     shap_values[i] = explainer.shap_values(X)

for i, (train_val_index, test_index, fold) in enumerate(mcv.splitter(X, ringnrs)):

    X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
    y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
    ringnr_train_val, ringnr_test = (
        ringnrs.iloc[train_val_index],
        ringnrs.iloc[test_index],
    )

    explainer = pltreeshap.PLTreeExplainer(mods[i], data = X_train_val)
    shap_values[i] = explainer.shap_values(X_test)
         


mean_shap = np.abs(shap_values).mean(axis = 0)

shap_df = pd.DataFrame(mean_shap, columns = X.columns)
shap_df.to_feather(mod_path / "shap.feather")
