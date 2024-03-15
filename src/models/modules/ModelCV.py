"""
Path: src/models/ModelCV.py
This file contains the ModelCV class and its variants
which is used for cross-validation of models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
import shutil
from ..utils import prep_data_before_train, Dataset, ModelConfig, searchspaces, prep_data_for_delta_method
from .ModelTrainer import ModelTrainer, INLATrainer, QuantileTrainer

class ModelCV:
    """
    MdodelCV class is used for cross-validation of models.
    use the run() method to run the cross-validation.
    Results are automatically saved in the project/models folder.
    """

    def __init__(
        self, data_path: Path, modelSettings: ModelConfig, n_splits: int = 10
    ) -> None:
        """
        Constructor for ModelCV class.
        param data_path: Path to the data, feather file.
        param modelSettings: ModelConfig object.
        param n_splits: Number of splits for cross-validation. default=10.
        """
        self.data_path = data_path
        self.modelSettings = modelSettings
        self.n_splits = n_splits
        self.results = pd.DataFrame(columns=["name", "phenotype", "fold", "corr", "model_id"])
        self.project_path = data_path.parents[2]

    def splitter(self, X, ringnrs):
        kf = GroupKFold(n_splits=self.n_splits)
        kfsplits = kf.split(X, groups=ringnrs)
        fold_names = [i for i in range(self.n_splits)]
        for i, (train, test) in enumerate(kfsplits):
            yield train, test, fold_names[i]

    def train_and_eval(self, dataset: Dataset):
        trainer = ModelTrainer(modelSettings=self.modelSettings, data=dataset, max_evals= self.modelSettings.hyp_settings["max_evals"])
        trainer.hypertrain()
        trainer.save(project_path=self.project_path)
        try:
            y_preds = trainer.bestModel.predict(dataset.X_test, iteration_range= (0, trainer.bestModel.best_iteration))
        except:
            y_preds = trainer.bestModel.predict(dataset.X_test) 
        self.corr = pearsonr(y_preds, dataset.y_test)[0]
        print(f"FOLD {dataset.fold} finished\t corr: {self.corr} ")

    def run(self):
        data = pd.read_feather(self.data_path)

        if self.modelSettings.procedure == "delta":
            X, y, ringnrs = prep_data_for_delta_method(
            data=data, phenotype=self.modelSettings.phenotype
            )
        else:
            X, y, ringnrs = prep_data_before_train(
                data=data, phenotype=self.modelSettings.phenotype
            )

        for _, (train_val_index, test_index, fold) in enumerate(
            self.splitter(X, ringnrs)
        ):
            X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
            y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
            ringnr_train_val, ringnr_test = (
                ringnrs.iloc[train_val_index],
                ringnrs.iloc[test_index],
            )
            if self.modelSettings.procedure == "two-step":
                X_train_val = X_train_val.drop(columns = ["hatchisland"])
                X_test = X_test.drop(columns= ["hatchisland"])

            dataset = Dataset(
                X_train_val=X_train_val,
                X_test=X_test,
                y_train_val=y_train_val,
                y_test=y_test,
                ringnr_train_val=ringnr_train_val,
                ringnr_test=ringnr_test,
                fold=fold,
            )
            self.train_and_eval(dataset)
            self.add_to_results(fold)
        self.save()

    def add_to_results(self, fold)-> None:
        if fold == 0 or fold == "outer" or fold == "island_0":
            self.results = pd.DataFrame(
                {
                    "name": self.modelSettings.name,
                    "phenotype": self.modelSettings.phenotype,
                    "fold": fold,
                    "corr": self.corr,
                    "model_id": self.modelSettings.model_id
                },
                index=[0],
            )
        else:
            self.results.loc[len(self.results.index)] = [
                self.modelSettings.name,
                self.modelSettings.phenotype,
                fold,
                self.corr,
                self.modelSettings.model_id
            ]


    def save(self):
        save_path = self.project_path / "models" / self.modelSettings.name
        shutil.copyfile(self.modelSettings.yaml_path, save_path / "config.yaml")

        old_results = pd.read_pickle(save_path.parent / "results.pkl")
        self.results = pd.concat([old_results, self.results], axis=0)
        self.results = self.results.reset_index(drop = True)
        self.results = self.results.drop_duplicates()
        self.results.to_pickle(save_path.parent / "results.pkl")


class ModelOuterInner(ModelCV):
    """
    Extends ModelCV class. Used for across population predictions.
    Train once on the outer population and test on the inner population.
    An train once on the inner population and test on the outer population.
    """

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
    """
    Extends ModelCV class. Used for across population predictions.
    Trains on all islands except one and tests on the left out island, repeats for all islands.
    """

    def __init__(
        self, data_path: Path, modelSettings: ModelConfig, n_splits: int = 10
    ) -> None:
        super().__init__(data_path, modelSettings, n_splits)

    def splitter(self, X, ringnrs):
        islands = X.hatchisland.unique()
        islands = islands[islands >= 5]
        for i in range(len(islands)):
            train_val_index = X.index[X.hatchisland != islands[i]]
            test_index = X.index[X.hatchisland == islands[i]]
            fold_name = "island_" + str(islands[i])
            yield train_val_index, test_index, fold_name


class ModelINLA(ModelCV):
    """
    Extends ModelCV class. Used when training INLA models (or R models)
    Saves the ringnrs for each fold in a feather file, so that they are available later.
    The config is saved, but the results must be saved in th R script.
    """

    def _init__(
        self, data_path: Path, modelSettings: ModelConfig, n_splits: int = 10
    ) -> None:
        super().__init__(data_path, modelSettings, n_splits)

    def run(self):
        data = pd.read_feather(self.data_path)
        X, y, ringnrs = prep_data_before_train(
            data=data, phenotype=self.modelSettings.phenotype
        )
        for _, (train_val_index, test_index, fold) in enumerate(
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
            trainer = INLATrainer(modelSettings=self.modelSettings, data=data)
            trainer.save(project_path=self.project_path)
        self.save()

    def save(self):
        save_path = self.project_path / "models" / self.modelSettings.name
        shutil.copyfile(self.project_path / "config.yaml", save_path / "config.yaml")

'''Functions for quantile regresssion (class is dynamically created in train_model script)'''

def train_and_eval_quantile(self, dataset: Dataset):
    trainer = QuantileTrainer(modelSettings=self.modelSettings, data=dataset)
    trainer.hypertrain()
    trainer.save(project_path=self.project_path)
    y_preds = trainer.bestModel.predict(dataset.X_test, iteration_range = (0, trainer.bestModel.best_iteration))
    self.corr_lower = pearsonr(y_preds[:, 0], dataset.y_test)[0]
    self.corr = pearsonr(y_preds[:, 1], dataset.y_test)[0]
    self.corr_upper = pearsonr(y_preds[:, 2], dataset.y_test)[0]

    print(
        f"FOLD {dataset.fold} finished, corr_lower: {self.corr_lower}\t corr:{self.corr}\t corr_upper:{self.corr_upper}"
    )


def add_to_results_quantile(self, fold)-> None:
    if fold == 0 or fold == "outer" or fold == "island_0":
        self.results = pd.DataFrame(
            {
                "name": self.modelSettings.name,
                "phenotype": self.modelSettings.phenotype,
                "fold": fold,
                "corr": self.corr,
                "model_id": self.modelSettings.model_id
            },
            index=[0],
        )
        self.results_quantile = pd.DataFrame(
            {
                "name": [self.modelSettings.name]*3,
                "phenotype": [self.modelSettings.phenotype]*3,
                "fold": [fold]*3,
                "corr": [self.corr, self.corr_lower, self.corr_upper],
                "quantile": [0.5, 0.25, 0.75],
                "model_id": [self.modelSettings.model_id]*3
            },
            index = [0,1,2],
        )
    else:
        self.results.loc[len(self.results.index)] = [
            self.modelSettings.name,
            self.modelSettings.phenotype,
            fold,
            self.corr,
            self.modelSettings.model_id
        ]
        corr_list = [self.corr, self.corr_lower, self.corr_upper]
        quantile_list = [0.5, 0.25, 0.75]
        for i in range(len(corr_list)):
            self.results_quantile.loc[len(self.results_quantile.index)] = [
                self.modelSettings.name,
                self.modelSettings.phenotype,
                fold,
                corr_list[i],
                quantile_list[i],
                self.modelSettings.model_id
            ]
 

def save_quantile(self):
    save_path = self.project_path / "models" / self.modelSettings.name
    shutil.copyfile(self.modelSettings.yaml_path, save_path / "config.yaml")

    old_results = pd.read_pickle(save_path.parent / "results.pkl")
    old_results_quantile = pd.read_pickle(save_path.parent / "results_quantile.pkl")

    self.results = pd.concat([old_results, self.results], axis=0)
    self.results_quantile = pd.concat([old_results_quantile, self.results_quantile], axis=0)
    
    self.results = self.results.reset_index(drop = True)
    self.results = self.results.drop_duplicates()

    self.results_quantile = self.results_quantile.reset_index(drop = True)
    self.results_quantile = self.results_quantile.drop_duplicates()

    self.results.to_pickle(save_path.parent / "results.pkl")
    self.results_quantile.to_pickle(save_path.parent / "results_quantile.pkl")


''' Benchagainst using the mean '''


def train_and_eval_mean(self, dataset: Dataset):
    y_mean = np.mean(dataset.y_train_val)
    y_preds = np.repeat(y_mean, len(dataset.y_test))
    self.corr = pearsonr(y_preds, dataset.y_test)[0]
    mse = np.mean(np.abs(y_preds - dataset.y_test))
    print(
        f"FOLD {dataset.fold} finished, mse :{mse}"
    )
    if dataset.fold == "inner" or dataset.fold == 9:
        import sys
        sys.exit()





