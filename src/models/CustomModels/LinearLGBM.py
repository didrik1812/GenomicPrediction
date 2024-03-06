import lightgbm as lgbm 
from pathlib import Path

class LinearLGBM:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.best_iteration = None

    def _create_data(self, X,y):
        data = lgbm.Dataset(X, label = y, params= {'linear_tree':True})
        return data


    def fit(self, X, y, **kwargs):
        data = self._create_data(X,y)
        if "eval_set" in kwargs.keys():
            X_val, y_val = kwargs["eval_set"][0]
            evalset = self._create_data(X_val, y_val)
            self._model = lgbm.train(self.kwargs, data, valid_sets= [evalset])
        else:
            self._model = lgbm.train(self.kwargs, data)

    def predict(self, X, **kwargs):
        return self._model.predict(X, num_iteration=self._model.best_iteration)

    def save_model(self, path:Path):
        self._model.save_model(path.parent / (path.stem + ".txt"))


