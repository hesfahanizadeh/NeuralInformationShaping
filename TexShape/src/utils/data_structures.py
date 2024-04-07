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
    utility_stats_network_model_path: Path = None
    privacy_stats_network_model_path: Path = None
    
@dataclass
class EncoderParams:
    encoder_hidden_sizes: list = field(default_factory=lambda: [512, 256, 128])
    num_enc_epochs: int = 10
    enc_save_dir_path: Path = None

@dataclass
class ExperimentParams:
    dataset_name: str
    # TODO: Use ENUM or dict
    experiment_type: str  # "utility+privacy"
    mine_args: MINE_Params
    encoder_params: EncoderParams
    experiment_date: str
    
    beta: float = 0.2   
    combination_type: str = "premise_only"
    
    @property
    def experiment_name(self) -> str:
        return f"{self.dataset_name}_{self.experiment_type}_{self.experiment_date}"

# @dataclass
# class HParams:
#     batch_size: int
#     learning_rate: float
#     epochs: int
#     num_classes: int
#     input_shape: tuple
#     dropout_rate: float

#     def __post_init__(self):
#         self.input_shape = (self.input_shape[0], self.input_shape[1], 1)

#     def __str__(self):
#         return f"batch_size: {self.batch_size}, learning_rate: {self.learning_rate}, epochs: {self.epochs}, num_classes: {self.num_classes}, input_shape: {self.input_shape}, dropout_rate: {self.dropout_rate}"
