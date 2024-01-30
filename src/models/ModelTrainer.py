"""
Path: src/models/ModelTrainer
This module contains the ModelTrainer class,
which is used to train a model with hyperparameter optimization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import pickle
from pathlib import Path
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from functools import partial
from sklearn.model_selection import GroupKFold, train_test_split, GroupShuffleSplit
from scipy.stats import pearsonr
import shutil
from .utils import prep_data_before_train, Dataset, ModelConfig


class ModelTrainer:
    def __init__(self, modelSettings: ModelConfig, data: Dataset, max_evals=30) -> None:
        self.modelSettings = modelSettings
        self.data = data
        self.max_evals = max_evals
        self.BestModel = None

    def hypertrain(self):
        trials = Trials()
        objective = partial(self.objective)
        best = fmin(
            objective,
            space=self.modelSettings.search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
        )
        self.bestModel = self.modelSettings.model(
            **{**self.modelSettings.fixed_params, **best}
        )
        self.bestModel.fit(self.data.X_train_val, self.data.y_train_val)

    def objective(self):
        merged_params = {
            **self.modelSettings.fixed_params,
            **self.modelSettings.train_params,
        }
        model = self.modelSettings.model(**merged_params)
        model.fit(self.data.X_train, self.data.y_train)
        y_pred = model.predict(self.data.X_val)
        mse = np.mean((y_pred - self.data.y_val) ** 2)
        return {"loss": mse, "status": STATUS_OK}

    def save(self, project_path: Path):
        path = project_path / "models" / self.modelSettings.name
        path.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.BestModel, open(path / f"{self.data.fold}", "wb"))


class INLAtrainer(ModelTrainer):
    def __init__(self, modelSettings: ModelConfig, data: Dataset) -> None:
        super().__init__(modelSettings, data)

    def hypertrain(self):
        pass

    def objective(self):
        pass

    def save(self, project_path: Path):
        pass
