'''
Path: src/models/controller.py
This file is the controller of the model training process.
'''
from .utils import (
    prep_data_before_train,
    handle_yaml_before_train,
    Dataset,
    ModelConfig,
)
from .ModelTrainer import ModelTrainer, ModelCV

class ModelController:
    def __init__(self) -> None:
        pass
    