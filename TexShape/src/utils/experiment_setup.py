from .data_structures import ExperimentParams, MINE_Params, EncoderParams, LogParams
from omegaconf import DictConfig
import datetime

def get_date() -> str:
    return datetime.datetime.now().strftime("%m_%d_%y")

def get_experiment_params(config: DictConfig) -> ExperimentParams:
    mine_params = MINE_Params(**config.experiment.mine_params)
    encoder_params = EncoderParams(**config.experiment.encoder_params)
    log_params = LogParams(**config.experiment.log_params)
    experiment_date = get_date()
    experiment_params = ExperimentParams(
        dataset_name=config.experiment.dataset_name,
        experiment_type=config.experiment.experiment_type,
        mine_params=mine_params,
        encoder_params=encoder_params,
        log_params=log_params,
        experiment_date=experiment_date,
        beta=config.experiment.beta,
        combination_type=config.experiment.combination_type
    )
    return experiment_params
