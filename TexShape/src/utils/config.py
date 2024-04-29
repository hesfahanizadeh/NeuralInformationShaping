"""
Functions and classes for configuration parameters.
Author: H. Kaan Kale
Email: hkaankale1@gmail.com
"""

from abc import ABC
from typing import Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from omegaconf import DictConfig


class TestType(Enum):
    """Test types."""

    RANDOM = "RANDOM"
    ORIGINAL = "ORIGINAL"
    TEXSHAPE = "TEXSHAPE"
    NOISE = "NOISE"
    QUANTIZATION = "QUANTIZATION"


class ExperimentType(Enum):
    """
    Experiment types.
    UTILITY: Utility only.
    UTILITY_PRIVACY: Utility and privacy.
    COMPRESSION: Compression only.
    COMPRESSION_PRIVACY: Compression and privacy.
    """

    UTILITY = "utility"
    UTILITY_PRIVACY = "utility+privacy"
    COMPRESSION = "compression"
    COMPRESSION_PRIVACY = "compression+privacy"


class DatasetName(Enum):
    """Dataset names."""

    SST2 = "sst2"
    MNLI = "mnli"
    CORONA = "corona"


class MNLICombinationType(Enum):
    """MNLI combination types."""

    CONCAT = "concat"
    JOIN = "join"
    PREMISE_ONLY = "premise_only"


@dataclass
class MineParams:
    """Mutual Information Neural Estimation Parameters."""

    utility_stats_network_model_name: str
    utility_stats_network_model_params: dict
    privacy_stats_network_model_name: str
    privacy_stats_network_model_params: dict
    use_prev_epochs_mi_model: bool
    mine_trainer_patience: int

    mine_batch_size: int = -1
    mine_epochs_privacy: int = 2000
    mine_epochs_utility: int = 2000


class DatasetParams(ABC):
    """Abstract Dataset Params."""

    dataset_loc: Union[Path, str]
    dataset_name: Union[DatasetName, str]


@dataclass
class SST2Params(DatasetParams):
    """SST2 Dataset Params"""

    dataset_loc: Path
    dataset_name: Union[DatasetName, str]

    def __post_init__(self):
        self.dataset_name: DatasetName = DatasetName.SST2
        if isinstance(self.dataset_loc, str):
            self.dataset_loc = Path(self.dataset_loc)


@dataclass
class CoronaParams(DatasetParams):
    """Corona Dataset Params"""

    dataset_loc: Path
    dataset_name: Union[DatasetName, str]

    def __post_init__(self):
        self.dataset_name: DatasetName = DatasetName.CORONA
        if isinstance(self.dataset_loc, str):
            self.dataset_loc = Path(self.dataset_loc)


@dataclass
class MNLIParams(DatasetParams):
    """MNLI Dataset Params"""

    dataset_loc: Path
    combination_type: MNLICombinationType
    dataset_name: Union[DatasetName, str]

    def __post_init__(self):
        self.dataset_name: DatasetName = DatasetName.MNLI

        self.combination_type = MNLICombinationType(self.combination_type)
        if isinstance(self.dataset_loc, str):
            self.dataset_loc = Path(self.dataset_loc)


@dataclass
class EncoderParams:
    """
    Encoder Model Params.
    :param encoder_model_name: Encoder model name.
    :param encoder_model_params: Encoder model parameters.
    :param num_enc_epochs: Number of encoder training epochs.
    :param encoder_learning_rate: Encoder learning rate.
    """

    encoder_model_name: str
    encoder_model_params: dict
    num_enc_epochs: int
    encoder_learning_rate: float


@dataclass
class ExperimentParams:
    """
    Experiment parameters.
    :param experiment_type: Experiment type.
    :param beta: Beta parameter.
    :param mine_params: Mutual Information Neural Estimation parameters.
    :param encoder_params: Encoder model parameters.
    :param dataset_params: Dataset parameters.

    :func __post_init__: Post initialization function.
    Type cast the experiment type, mine params, encoder params, and dataset params.
    :raises ValueError: If the dataset name is invalid.
    """

    experiment_type: Union[ExperimentType, str]  # "utility+privacy"
    beta: float
    mine_params: Union[MineParams, DictConfig]
    encoder_params: Union[EncoderParams, DictConfig]
    dataset_params: Union[DatasetParams, DictConfig]

    def __post_init__(self):
        self.experiment_type = ExperimentType(self.experiment_type)
        self.mine_params = MineParams(**self.mine_params)
        self.encoder_params = EncoderParams(**self.encoder_params)

        dataset_name = self.dataset_params.dataset_name
        if dataset_name == DatasetName.SST2.value:
            self.dataset_params = SST2Params(**self.dataset_params)
        elif dataset_name == DatasetName.MNLI.value:
            self.dataset_params = MNLIParams(**self.dataset_params)
        elif dataset_name == DatasetName.CORONA.value:
            self.dataset_params = CoronaParams(**self.dataset_params)
        else:
            raise ValueError("Invalid dataset name")


def set_include_privacy(experiment_type: str) -> bool:
    """Check if privacy is included in the experiment."""
    if (
        experiment_type == ExperimentType.UTILITY_PRIVACY
        or experiment_type == ExperimentType.COMPRESSION_PRIVACY
    ):
        return True
    return False


def load_experiment_params(config: DictConfig) -> ExperimentParams:
    """Load the experiment parameters."""
    experiment_params = ExperimentParams(
        experiment_type=config.simulation.experiment_type,
        mine_params=config.simulation.mine,
        dataset_params=config.dataset,
        encoder_params=config.encoder,
        beta=config.simulation.beta,
    )
    return experiment_params
