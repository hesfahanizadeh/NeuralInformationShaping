from src.models.models_to_train import Encoder, MI_CalculatorModel
from src.models import models_to_train

def create_encoder_model(*, model_name: str, model_params: dict) -> Encoder:
    return vars(models_to_train)[model_name](**model_params)

def create_mi_calculator_model(*, model_name:str, model_params:dict) -> MI_CalculatorModel:
    return vars(models_to_train)[model_name](**model_params)