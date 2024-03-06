from hyperopt import hp

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

xgboost_linear_space = {
    "lambda": hp.loguniform("lambda", -8, 2),
    "alpha": hp.loguniform("alpha", -8, 2),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "eta": hp.loguniform("learning_rate", -7, 0),
    # top_k can only be used with greedy or thrifty feature selector
    # "top_k": hp.randint("top_k", int(1e3), int(3e4) )
}

elasticnet_space ={
        "alpha": hp.loguniform("alpha", -4, 0),
        "l1_ratio": hp.loguniform("l1_ratio", -4, 0),

}

linearresidtree_space = {
    **xgboost_space,
    **xgboost_linear_space
}

search_spaces = {
    "xgboost_space": xgboost_space,
    "lightgbm_space": LGBM_space,
    "catboost_space": catboost_space,
    "GBM_space": GBM_space,
    "xgboost_linear_space": xgboost_linear_space,
    "linearresidtree_space": linearresidtree_space,
    "elasticnet_space": elasticnet_space,
    "linearlgbm_space":linearlgbm_space,
}
