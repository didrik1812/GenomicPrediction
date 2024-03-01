import pandas as pd
import numpy as np
from ..utils import searchspaces
from xgboost import XGBRegressor, DMatrix
from pathlib import Path

valid_linear_params = list(searchspaces.xgboost_linear_space.keys()) + [
    "early_stopping_rounds",
    "objective",
    "quantile_alpha",
]
valid_tree_params = list(searchspaces.xgboost_space.keys()) + [
    "early_stopping_rounds",
    "objective",
    "quantile_alpha",
]


class HuberContamination:
    def __init__(self, **kwargs) -> None:
        linear_model_params = {
            k: kwargs[k] for k in kwargs.keys() if k in valid_linear_params
        }
        tree_model_params = {
            k: kwargs[k] for k in kwargs.keys() if k in valid_tree_params
        }
        self.linear_model = XGBRegressor(**linear_model_params, booster="gblinear")
        self.tree_model = XGBRegressor(**tree_model_params, booster="gbtree")
        self.best_iteration = None
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        self.linear_model.fit(X, y, **kwargs)
        self.tree_model.fit(X, y, **kwargs)
        self.calc_mean_leaf_values(X)

    def calc_mean_leaf_values(self, X:pd.DataFrame)->None:
        # To acsess leaf-indexes, we must go via the booster api
        Xarray = X.to_numpy()
        dtrain = DMatrix(X)
        booster = self.tree_model.get_booster()
        # get leafindexes, shape is (observation, leafindex)
        leafindex= booster.predict(dtrain, pred_leaf = True)
        # collect the sum of feature-vectors for each leaf-index for all weak learners
        self._obs_leaf = [{k:[] for k in np.unique(leafindex[:,i])} for i in range(leafindex.shape[1])]
        self._mean_obs_leaf = [0]*leafindex.shape[1]
        # self._std_obs_leaf = [0]*leafindex.shape[1]
        # self._obs_leaf_standardized = [{k:[] for k in np.unique(leafindex)}]*leafindex.shape[1]
        # Loop through all weak learners
        for i in range(leafindex.shape[1]):
            # loop through all observations and corresponding leafindex
            for j in range(leafindex.shape[0]):
                obs = Xarray[j]
                leaf = leafindex[j]
                # Add observation to leafindex
                self._obs_leaf[i][leaf[i]] += [obs] 
            # calculate mean feature-vectors of weak learner
            self._mean_obs_leaf[i] = {k: np.mean(self._obs_leaf[i][k], axis = 0) for k in self._obs_leaf[i].keys()}
            # self._std_obs_leaf[i] = {k: np.std(self._obs_leaf[i][k], axis = 0) for k in self._obs_leaf[i].keys()}
            # standardizee the feature-vectors 
            # std_lambda = lambda x,k: (x - self._mean_obs_leaf[i][k])/self._std_obs_leaf[i][k]
            # apply_std_lambda = lambda L,k: [std_lambda(l,k) for l in L] 
            # self._obs_leaf_standardized[i] = {k:apply_std_lambda(self._obs_leaf[i][k],k) for k in self._obs_leaf[i].keys()}


    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        kwargs_linear = kwargs
        kwargs_tree = kwargs
        if "iteration_range" in kwargs.keys():
            kwargs_linear["iteration_range"] = (0, self.linear_model.best_iteration)
            kwargs_tree["iteration_range"] = (0, self.tree_model.best_iteration)

        eps = self.get_contamination_rates(X)

        return eps * self.tree_model.predict(X, **kwargs_tree) + (1 - eps) * self.linear_model.predict(
            X, **kwargs_linear
        )

    def get_contamination_rates(self, X: pd.DataFrame) -> np.ndarray:
        booster = self.tree_model.get_booster() 
        dtest = DMatrix(X)
        leafindex = booster.predict(dtest, pred_leaf = True)
        # Calculate "distance" for each observation
        distance = np.zeros(X.shape[0])
        # Loop through new features
        Xarray = X.to_numpy()
        for i in range(leafindex.shape[0]):
            obs = Xarray[i]
            leaf = leafindex[i]
            for j in range(leafindex.shape[1]):
                mean_feature = self._mean_obs_leaf[j][leaf[j]]
                # std_feature = self._std_obs_leaf[j][leaf[j]]
                # Sum of standardized observations
                # obs_standardized = (obs - mean_feature)/std_feature
                # standardized_observations = self._obs_leaf_standardized[j][leaf[j]] + [obs_standardized]
                # calculate softmax compared to training obs
                # distance[i] += np.exp(np.sum(obs_standardized))/np.sum(np.exp(np.sum(standardized_observations,axis = 1))) 
                # Use cosine distance
                part = mean_feature.dot(obs)/(np.linalg.norm(mean_feature)*np.linalg.norm(obs))
                distance[i] += 1-part
        # take the mean across all learners 
        distance = distance * 1/leafindex.shape[1]
        # The closer to one distance is, the more different is the new obs from the training feature
        # therefore subtract it to one to act as tree-weight
        return 1-distance

    def save_model(self, save_path:Path):
        save_dir = (save_path.parent / save_path.stem)
        save_dir.mkdir(parents = True, exist_ok=True)
        self.linear_model.save_model(save_dir / "linear_booster.json")
        self.tree_model.save_model(save_dir / "tree_booster.json")
