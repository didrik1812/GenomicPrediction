"""
Path: src/models/ModelCV.py
This file contains the ModelCV class and its variants
which is used for cross-validation of models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split, GroupShuffleSplit
from scipy.stats import pearsonr
import shutil
from .utils import prep_data_before_train, Dataset, ModelConfig
from .ModelTrainer import ModelTrainer


class ModelCV:
    def __init__(
        self, data_path: Path, modelSettings: ModelConfig, n_splits: int = 10
    ) -> None:
        self.data_path = data_path
        self.modelSettings = modelSettings
        self.n_splits = n_splits
        self.results = pd.DataFrame(columns=["name", "phenotype", "fold", "corr"])
        self.project_path = data_path.parent.parent

    def splitter(self, X, ringnrs):
        kf = GroupKFold(n_splits=self.n_splits)
        kfsplits = kf.split(X, groups=ringnrs)
        fold_names = [i for i in range(self.n_splits)]
        for i, splits in enumerate(kfsplits):
            yield splits, fold_names[i]

    def run(self):
        data = pd.read_feather(self.data_path)
        X, y, ringnrs = prep_data_before_train(
            data=data, phenotype=self.modelSettings.phenotype
        )
        for i, (train_val_index, test_index, fold) in enumerate(
            self.splitter(X, ringnrs)
        ):
            X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
            y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
            ringnr_train_val, ringnr_test = (
                ringnrs.iloc[train_val_index],
                ringnrs.iloc[test_index],
            )

            data = Dataset(
                X_train_val=X_train_val,
                X_test=X_test,
                y_train_val=y_train_val,
                y_test=y_test,
                ringnr_train_val=ringnr_train_val,
                ringnr_test=ringnr_test,
                fold=fold,
            )
            trainer = ModelTrainer(modelSettings=self.modelSettings, data=data)
            trainer.hypertrain()
            trainer.save(project_path=self.project_path)
            y_preds = trainer.bestModel.predict(X_test)
            corr = pearsonr(y_preds, y_test)[0]
            if fold == 0:
                self.results = pd.DataFrame(
                    {
                        "name": self.modelSettings.name,
                        "phenotype": self.modelSettings.phenotype,
                        "fold": fold,
                        "corr": corr,
                    },
                    index=[0],
                )
            else:
                self.results.loc[len(self.results.index)] = [
                    self.modelSettings.name,
                    self.modelSettings.phenotype,
                    fold,
                    corr,
                ]
            self.save()

    def save(self):
        save_path = self.project_path / "models" / self.modelSettings.name
        shutil.copyfile(self.project_path / "config.yaml", save_path / "config.yaml")

        old_results = pd.read_pickle(save_path.parent / "results.pkl")
        self.results = pd.concat([old_results, self.results], axis=0)
        self.results.to_pickle(save_path.parent / "results.pkl")


class ModelOuterInner(ModelCV):
    def __init__(
        self, data_path: Path, modelSettings: ModelConfig, n_splits: int = 10
    ) -> None:
        super().__init__(data_path, modelSettings, n_splits)

    def splitter(self, X, ringnrs):
        outer_indexes = X.index[X.hatchisland.isin([22, 23, 24])]
        inner_indexes = X.index[~X.hatchisland.isin([22, 23, 24])]
        indexes = [outer_indexes, inner_indexes]
        fold_names = ["outer", "inner"]
        for i in range(2):
            train_val_index, test_index = indexes[i], indexes[1 - i]
            yield train_val_index, test_index, fold_names[i]


class ModelAcrossIsland(ModelCV):
    def __init__(
        self, data_path: Path, modelSettings: ModelConfig, n_splits: int = 10
    ) -> None:
        super().__init__(data_path, modelSettings, n_splits)

    def splitter(self, X, ringnrs):
        islands = X.hatchisland.unique()
        for i in range(len(islands)):
            train_val_index = X.index[X.hatchisland != islands[i]]
            test_index = X.index[X.hatchisland == islands[i]]
            fold_name = "island_i" + islands[i]
            yield train_val_index, test_index, fold_name
