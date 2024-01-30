import pandas as pd
import pickle
from xgboost import XGBRegressor
import numpy as np
from pathlib import Path
from src.models.utils import prep_data_before_train
from scipy.stats import pearsonr

np.random.seed(42)

MODEL_TO_FETCH = "xgboostMasssLinearEGA"
# MODEL_TO_FETCH = "xgboostMasssEGA"
FOLD = "fold_outer.pkl"
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / MODEL_TO_FETCH/FOLD
PHENOTYPE = "mass"
DATA_PATH = PROJECT_DIR / "data"/"processed"/f"{PHENOTYPE}EG.feather"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

data = pd.read_feather(DATA_PATH)
X, Y, ringnrs = prep_data_before_train(data, PHENOTYPE)


outer_indexes = X.index[X.hatchisland.isin([22, 23, 24])]
X_outer, X_inner = X.loc[outer_indexes], X.drop(outer_indexes, errors="ignore")
Y_outer, Y_inner = Y.loc[outer_indexes], Y.drop(outer_indexes, errors="ignore")

model_new = XGBRegressor(
    booster="gblinear",
    reg_alpha=0.1,
    reg_lambda=0.1,
    learning_rate=0.001,
    n_estimators=150
)
model_new.fit(X_outer, Y_outer)

Y_preds = model_new.predict(X_inner)
corr = pearsonr(Y_inner, Y_preds)[0]
print("MIN-MAX")
print(np.min(Y_outer), np.max(Y_outer))
print(np.min(Y_inner), np.max(Y_inner))
print(np.min(Y_preds), np.max(Y_preds))
print("Corr")
print(corr)
print("COEFS")
print(pd.DataFrame(model.coef_).describe())

results_df = pd.read_pickle(PROJECT_DIR / "models" / "results.pkl")
across_df = results_df[results_df.fold.isin(["outer", "inner"])]
print(across_df)
