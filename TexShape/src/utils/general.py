from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from pytorch_lightning import seed_everything
import numpy as np
import datetime
from abc import ABC
from typing import Union
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import DictConfig


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

class DatasetParams(ABC):
    dataset_loc: Union[Path, str]
    dataset_name: str


@dataclass
class SST2_Params(DatasetParams):
    dataset_name: str
    sent_len_threshold: int
    train_test_split_ratio: float
    dataset_loc: Path

    def __post_init__(self):
        if isinstance(self.dataset_loc, str):
            self.dataset_loc = Path(self.dataset_loc)


@dataclass
class MNLI_Params(DatasetParams):
    dataset_name: str
    dataset_loc: Path
    combination_type: str

    def __post_init__(self):
        if isinstance(self.dataset_loc, str):
            self.dataset_loc = Path(self.dataset_loc)

@dataclass
class EncoderParams:
    encoder_model_name: str
    encoder_model_params: dict
    num_enc_epochs: int = 10
    


@dataclass
class LogParams:
    log_dir_path: Path = field(default="")
    log_file_path: Path = field(default="")

    def __post_init__(self):
        self.log_dir_path = Path(self.log_dir_path)
        self.log_file_path = Path(self.log_file_path)


@dataclass
class ExperimentParams:
    dataset_name: str
    # TODO: Use ENUM or dict
    experiment_type: str  # "utility+privacy"
    beta: float
    mine_params: MINE_Params
    encoder_params: EncoderParams


# ROC and AUC functions # TODO Optimize this function
def get_roc_auc(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model = model.to(device)
    model = model.eval()

    # Get the predictions and labels
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(outputs.cpu().numpy()[:, 1])

    roc = roc_curve(all_labels, all_confidences)

    # Get the AUC
    auc_score = auc(roc[0], roc[1])
    return roc, auc_score


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed)


def configure_torch_backend():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_include_privacy(experiment_type: str) -> bool:
    if experiment_type == "utility+privacy" or experiment_type == "compression+filter":
        return True
    return False


def get_date() -> str:
    return datetime.datetime.now().strftime("%m_%d_%y")


def load_experiment_params(config: DictConfig) -> ExperimentParams:
    mine_params = MINE_Params(**config.simulation.mine)
    encoder_params = EncoderParams(**config.encoder)
    experiment_params = ExperimentParams(
        dataset_name=config.dataset.dataset_name,
        experiment_type=config.simulation.experiment_type,
        mine_params=mine_params,
        encoder_params=encoder_params,
        beta=config.simulation.beta,
    )
    return experiment_params


def load_log_params(config: DictConfig) -> LogParams:
    log_params = LogParams(**config.simulation.log)
    return log_params
