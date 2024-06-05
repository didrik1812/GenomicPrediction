'''
searchspaces.py
Here the searchspaces used in the hyperparametertuning is defined
'''

from hyperopt import hp
import numpy as np
import scipy.stats as st

#################################################################################################################
# SEARCH SPACES FROM PROJECT THESIS (BAYESIAN OPTIMIZATION)
# NOT USED IN THE MASTER THESIS
GBM_space = {
    "learning_rate": hp.loguniform("learning_rate", -3, -1),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "max_depth": hp.randint("max_depth", 5, 30),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.45),
}


catboost_space = {
    "learning_rate": hp.loguniform("learning_rate", -7, 0),
    "random_strength": hp.uniform("random_strength", 0, 20),
    "l2_leaf_reg": hp.loguniform("l2_leaf_reg", 1, 10),
    "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
    "leaf_estimation_iterations": hp.randint("leaf_estimation_iterations", 1, 10),
}


xgboost_space = {
    "max_depth": hp.randint("max_depth", 2, 10),
    "alpha": hp.loguniform("alpha", -8, 2),
    "lambda": hp.loguniform("lambda", -8, 2),
    "min_child_weight": hp.loguniform("min_child_weight", -8, 5),
    "eta": hp.loguniform("eta", -7, 0),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
    "gamma": hp.loguniform("gamma", -8, 2),
}


LGBM_space = {
    "learning_rate": hp.loguniform("learning_rate", -7, 0),
    "num_leaves": hp.randint("num_leaves", 10, 10000),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "subsample": hp.uniform("subsample ", 0.5, 1.0),
    "min_sum_hessian_in_leaf": hp.loguniform("min_sum_hessian_in_leaf", -15, 5),
    "min_data_in_leaf": hp.randint("min_data_in_leaf", 1,  500 ),
    "reg_alpha": hp.loguniform("reg_alpha", -8, 2),
    "reg_lambda": hp.loguniform("reg_lambda", -8, 2),
    "max_depth": hp.randint("max_depth", 15, 10000),
    "n_estimators": hp.randint("n_estimators", 20, 205),
}

linearlgbm_space = {
    "learning_rate": hp.loguniform("learning_rate", -7, 0),
    "num_leaves": hp.randint("num_leaves", 10, 100),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "subsample": hp.uniform("subsample ", 0.5, 1.0),
    "min_sum_hessian_in_leaf": hp.loguniform("min_sum_hessian_in_leaf", -10, 2),
    "min_data_in_leaf": hp.randint("min_data_in_leaf", 1,  100 ),
    "reg_alpha": hp.loguniform("reg_alpha", -8, 2),
    "reg_lambda": hp.loguniform("reg_lambda", -8, 2),
    "max_depth": hp.randint("max_depth", 15, 500),
    "n_estimators": hp.randint("n_estimators", 20, 205),
}


#################################################################################################################

# SEARCH SPACES USED IN MASTER THESIS, This uses randomized search

lgbm_space = {
    'n_estimators': np.arange(300, 600, 20),#[ 150, 200, 300, ],
    'learning_rate': np.arange(0.01, 0.061, 0.005),# [0.03, 0.05, 0.07, ],
    'max_depth': np.arange(2, 25),# [5, 7, 10, 15, 20,],
    'min_child_weight': np.arange(0.02, 0.1, 0.01),#[  0.1, 0.2, ],
    'min_child_samples': [ 30, 40, 50, 60, 80, 100, 120, 150, 170, 200, 300, 500, 700, ], 
    'reg_lambda': [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1,  ],
    'reg_alpha':  [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, ],
    'linear_tree': [True, ],
    'subsample': np.arange(0.4, 0.901, 0.05),#[0.3, 0.5,  0.8],
    'subsample_freq': [1],
    'colsample_bytree': np.arange(0.2, 0.81, 0.05), #[0.5, 0.8, ],#0.3, 0.5, 0.8],
    'colsample_bynode': np.arange(0.2, 1.01, 0.05), #[0.5, 0.8, ],#0.3, 0.5, 0.8],
    'linear_lambda': [1e-3, 3e-3, 1e-2, 3e-2, 0.1,],
    'min_data_per_group': [10, 20, 50, 100],
    'verbose': [-1],
}


xgboost_rand_space = {
    'n_estimators': st.randint(100,500),
    'learning_rate': st.loguniform(0.001, 0.1),
    'min_child_weight': st.randint(1, 15),
    'alpha': st.loguniform(1e-5, 1e-1),
    'lambda':st.loguniform(1e-5, 1e-1),
    'subsample': st.uniform(loc = 0.4, scale = 0.5),
    'subsample_freq': [1],
    'colsample_bytree':st.uniform(loc = 0.2, scale = 0.6),
    'colsample_bynode': st.uniform(loc = 0.2, scale = 0.7),
    'gamma': st.loguniform(1e-3, 1e-1),
    'max_depth': st.randint(1, 10),
    'verbosity': [0],
}



xgboost_linear_rand_space = {
     'lambda':st.loguniform(1e-5, 1e-1),
     'alpha': st.loguniform(1e-5, 1e-1),
     'n_estimators': st.randint(20,500),
     'learning_rate': st.loguniform(0.001, 0.1),

}


catboost_rand_space = {
    "learning_rate": st.loguniform(0.0001, 0.1),
    "random_strength": st.randint(0,20),
    "l2_leaf_reg": st.loguniform(0.1, 40),
    "bagging_temperature": st.uniform(0.1, 1),
    "leaf_estimation_iterations": st.randint(1,10),
    "iterations": st.randint(100, 800),
    "depth": st.randint(1,10)
}

# Collect all spaces in one dict
search_spaces = {
    "xgboost_space": xgboost_space,
    "lightgbm_space": LGBM_space,
    "catboost_space": catboost_space,
    "GBM_space": GBM_space,
    "linearlgbm_space":linearlgbm_space,
    "lgbm_space": lgbm_space,
    "xgboost_rand_space" : xgboost_rand_space, 
    "xgboost_linear_rand_space" : xgboost_linear_rand_space, 
    "catboost_rand_space" : catboost_rand_space, 
}
