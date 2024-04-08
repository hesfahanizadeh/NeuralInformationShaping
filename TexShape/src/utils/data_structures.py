from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MINE_Params:
    utility_stats_network_model_name: str
    utility_stats_network_model_params: dict
    privacy_stats_network_model_name: str
    privacy_stats_network_model_params: dict
    use_prev_epochs_mi_model: bool

    mine_batch_size: int = -1
    mine_epochs_privacy: int = 2000
    mine_epochs_utility: int = 2000
    # Use default factory function to avoid mutable default arguments
    # if string passed then convert to Path object
    utility_stats_network_model_path: Path = field(default_factory=Path)
    privacy_stats_network_model_path: Path = field(default_factory=Path)
    
@dataclass
class EncoderParams:
    encoder_model_name: str
    encoder_model_params: dict
    num_enc_epochs: int = 10
    enc_save_dir_path: Path = field(default_factory=Path)

@dataclass
class LogParams:
    log_dir_path: Path = field(default_factory=Path)
    log_file_path: Path = field(default_factory=Path)
    
@dataclass
class ExperimentParams:
    dataset_name: str
    # TODO: Use ENUM or dict
    experiment_type: str  # "utility+privacy"
    mine_params: MINE_Params
    encoder_params: EncoderParams
    log_params: LogParams
    experiment_date: str
    
    beta: float = 0.2   
    combination_type: str = "premise_only"
    
    @property
    def experiment_name(self) -> str:
        return f"{self.dataset_name}_{self.experiment_type}_{self.experiment_date}"
