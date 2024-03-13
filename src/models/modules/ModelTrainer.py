"""
Path: src/models/ModelTrainer
This module contains the ModelTrainer class,
which is used to train a model with hyperparameter optimization.
"""
import numpy as np
from pathlib import Path
from hyperopt import fmin, tpe, Trials, STATUS_OK
from functools import partial
from ..utils import Dataset, ModelConfig
from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit

class ModelTrainer:
    """
    ModelTrainer class is used for training a model with hyperparameter optimization.
    use the hypertrain() method to run the hyperparameter optimization to find the best model.
    Save the best model with the save() method.
    """

    def __init__(self, modelSettings: ModelConfig, data: Dataset, max_evals=30) -> None:
        """
        Constructor for ModelTrainer class.
        param modelSettings: ModelConfig object.
        param data: Dataset object.
        param max_evals: Number of iterations for hyperparameter optimization. default=30.
        """
        self.modelSettings = modelSettings
        self.data = data
        self.max_evals = max_evals
        self.bestModel = None

    def splitter(self):
        kf = GroupShuffleSplit(n_splits=5)
        for (train_inds, val_inds) in kf.split(self.data.X_train_val, self.data.y_train_val, groups=self.data.ringnr_train_val):
            yield train_inds, val_inds

    def hypertrain(self):
        current_model = self.modelSettings.model(**self.modelSettings.fixed_params)
        hyp_cv_split = self.splitter()
        self.bestModel = RandomizedSearchCV(current_model, self.modelSettings.search_space, n_jobs = 5, n_iter = 10, cv = hyp_cv_split, verbose = 2)
        self.bestModel.fit(self.data.X_train_val, self.data.y_train_val)

    def save(self, project_path: Path):
        path = project_path / "models" / self.modelSettings.name
        path.mkdir(parents=True, exist_ok=True)
        try:
            self.bestModel.best_estimator_.save_model(path / f"{self.data.fold}.json")
        except:
            import pickle
            pickle.dump(self.bestModel.best_estimator_, open(path / f"{self.data.fold}.pkl", "wb"))


class INLATrainer(ModelTrainer):
    """
    INLATrainer class is an extension of the ModelTrainer class
    Only preps data for INLA models (to be run in R).
    Ensures that R-models are trained on equal folds as the python models.
    """

    def __init__(self, modelSettings: ModelConfig, data: Dataset) -> None:
        super().__init__(modelSettings, data)

    def hypertrain(self):
        pass

    def objective(self, params = None):
        pass

    def save(self, project_path: Path):
        save_path = project_path / "data" / "interim"
        self.data.ringnr_train_val.to_frame().reset_index().to_feather(
            save_path / f"ringnr_train_{self.data.fold}.feather"
        )
        self.data.ringnr_test.to_frame().reset_index().to_feather(
            save_path / f"ringnr_test_{self.data.fold}.feather"
        )


class QuantileTrainer(ModelTrainer):
    def __init__(self, modelSettings: ModelConfig, data: Dataset) -> None:
        super().__init__(modelSettings, data)

    def objective(self, params):
        merged_params = {
            **self.modelSettings.fixed_params,
            **params
        }
        model = self.modelSettings.model(**merged_params)
        model.fit(self.data.X_train, self.data.y_train, eval_set = [(self.data.X_val, self.data.y_val)], verbose = False)
        y_pred = model.predict(self.data.X_val, iteration_range = (0, model.best_iteration))[:,1] # access the median
        mse = np.mean((y_pred - self.data.y_val) ** 2)
        return {"loss": mse, "status": STATUS_OK}

