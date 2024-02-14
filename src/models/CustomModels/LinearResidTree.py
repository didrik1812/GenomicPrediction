import pandas as pd
import numpy as np
from ..utils import searchspaces
from xgboost import XGBRegressor
from pathlib import Path

valid_linear_params = list(searchspaces.xgboost_linear_space.keys()) + ["early_stopping_rounds", "objective","quantile_alpha" ]
valid_tree_params = list(searchspaces.xgboost_space.keys()) + ["early_stopping_rounds", "objective","quantile_alpha" ]

class LinearResidTree:
    '''
    Mix of linear booster and tree boster
    Idea: Trees cannot extrapolate but are good at nonlinear interactions, therefore use the linear booster
            to predict the main phenotype, and then the tree booster for the residuals.
    Made to mimick the behavior of the xgboost Sklearn api
    '''
    def __init__(self,  **kwargs) -> None:
        linear_model_params = {k:kwargs[k] for k in kwargs.keys() if  k in valid_linear_params} 
        tree_model_params = {k:kwargs[k] for k in kwargs.keys() if k in valid_tree_params} 
        self.linear_model = XGBRegressor(**linear_model_params, booster = "gblinear")
        self.tree_model = XGBRegressor(**tree_model_params, booster = "gbtree")
        self.best_iteration = None

    def fit(self, X:pd.DataFrame, y:pd.DataFrame, **kwargs)-> None:
        self.linear_model.fit(X,y, **kwargs)
        residuals = y - self.linear_model.predict(X)  
        # Try to avoid small numbers
        residuals[residuals <= 1e-6] = 0.0
        self.tree_model.fit(X, residuals, **kwargs)
        # Just need to set it to something to awoid problems
        self.best_iteration = self.linear_model.best_iteration

    def predict(self, X:pd.DataFrame, **kwargs)-> np.ndarray:
        if "iteration_range" in kwargs.keys():
            kwargs_linear = kwargs
            kwargs_tree = kwargs
            kwargs_linear["iteration_range"] = (0,self.linear_model.best_iteration)
            kwargs_tree["iteration_range"] = (0,self.tree_model.best_iteration)

            return self.linear_model.predict(X, **kwargs_linear) + self.tree_model.predict(X, **kwargs_tree)

        else:
            return self.linear_model.predict(X, **kwargs) + self.tree_model.predict(X, **kwargs)

    def save_model(self, save_path:Path):
        save_dir = (save_path.parent / save_path.stem)
        save_dir.mkdir(parents = True, exist_ok=True)
        self.linear_model.save_model(save_dir / "linear_booster.json")
        self.tree_model.save_model(save_dir / "tree_booster.json")
