"""
Test the models on the test dataset.
Author: H. Kaan Kale
Email: hkaankale1@gmail.com
"""

from pathlib import Path
from typing import Tuple, Union
import logging

from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils.general import (
    set_seed,
    configure_torch_backend,
)
from src.utils.config import (
    ExperimentParams,
    ExperimentType,
    DatasetName,
    MNLIParams,
    MNLICombinationType,
    load_experiment_params,
)
from src.test.utility_privacy import UtilityPrivacyTester
from src.test.compression import CompressionTester
from src.test.utils import TestParams


def find_experiment_config(
    *,
    experiments_dir: Union[Path, str],
    experiment_date: str = None,
    test_config: DictConfig = None,
) -> Tuple[ExperimentParams, Path]:
    """
    Find the experiment config file in the experiments directory.
    If experiment_date is provided, load the experiment config file from the experiments directory.
    If experiment_date is not provided,
    find the experiment with the same params as the test_experiment_config.

    :params test_config: DictConfig, the test config file.
    :params experiments_dir: Path, the experiments directory path.
    :params experiment_date: str, the experiment date to load the experiment config file.
    :returns: ExperimentParams, the experiment params of the experiment config file.
    Path, Experiment directory path
    :raises: ValueError, if the experiment with the same params
    as the test_experiment_config is not found.
    """

    # If experiment_date is provided, load the experiment config file from the experiments directory
    if experiment_date is not None:
        # Find the experiment config file in the experiments directory
        experiment_params, experiment_dir = find_config_in_experiments_by_date(
            experiment_date=experiment_date, experiments_dir=experiments_dir
        )
        return experiment_params, experiment_dir
    # If experiment_date is not provided,
    # find the experiment with the same params as the test_experiment_config
    if test_config is not None:
        experiment_params, experiment_dir = find_config_in_experiments_by_config(
            config=test_config, experiments_dir=experiments_dir
        )
        return experiment_params, experiment_dir
    else:
        raise ValueError(
            f"Experiment with params {test_config} not found in "
            f"{experiments_dir} and experiment_date {experiment_date}"
        )


def find_config_in_experiments_by_date(
    experiment_date: str, experiments_dir: Union[Path, str]
) -> Tuple[ExperimentParams, Path]:
    """
    Find the experiment config file in the experiments directory by date.

    :params experiment_date: str, the experiment date.
    :params experiments_dir: Path, the experiments directory path.
    :returns: ExperimentParams, the experiment params of the experiment config file.
    Path, Experiment directory path
    """
    if isinstance(experiments_dir, str):
        experiments_dir = Path(experiments_dir)

    experiment_dir = experiments_dir / experiment_date
    experiment_config_file = experiment_dir / ".hydra" / "config.yaml"

    if not experiment_config_file.exists():
        raise FileNotFoundError(
            f"Experiment config file not found in {experiment_config_file}"
        )

    experiment_config = OmegaConf.load(experiment_config_file)
    experiment_params = load_experiment_params(experiment_config)
    return experiment_params, experiment_dir


def find_config_in_experiments_by_config(
    config: Union[DictConfig, ExperimentParams, Path], experiments_dir: Union[Path, str]
) -> Tuple[ExperimentParams, Path]:
    """
    Find the experiment config file in the experiments directory.

    :params config: DictConfig, ExperimentParams, Path,
    the config file to find in the experiments directory,
    DictConfig or ExperimentParams to find in the experiments directory
    :returns: ExperimentParams, the experiment params of the experiment config file.
    Path, Experiment directory path
    """
    if isinstance(config, Path):
        config = OmegaConf.load(config)

    if isinstance(config, DictConfig):
        config: ExperimentParams = load_experiment_params(config)

    if isinstance(experiments_dir, str):
        experiments_dir = Path(experiments_dir)

    # Sort experiments dir according to the date and hour
    experiments_dir = sorted(
        experiments_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
    )

    for experiment_dir in experiments_dir:
        experiment_config_file = experiment_dir / ".hydra" / "config.yaml"
        # Load the experiment config as DictConfig
        experiment_config = OmegaConf.load(experiment_config_file)
        experiment_params = load_experiment_params(experiment_config)

        if experiment_params == config:
            return experiment_params, experiment_dir

    raise ValueError(f"Experiment with params {config} not found in {experiments_dir}")


@hydra.main(config_path="configs", config_name="test_config", version_base="1.2")
def main(config: DictConfig) -> None:
    """Main function to test the models on the test dataset."""
    # Configure the logger
    logging.basicConfig(level=logging.DEBUG)

    # Number of epochs for training the test classifiers
    load_from_epoch: int = 8

    # Set the seed and configure the torch backend
    seed: int = 42
    set_seed(seed)
    configure_torch_backend()

    # Test config parameters
    batch_size: int = config.batch_size
    device_idx: int = config.device_idx
    experiments_dir: str = config.experiments_dir
    test_experiment_config = config.experiment
    test_params = TestParams(max_epoch=-1, batch_size=batch_size, device_idx=device_idx)

    # Load experiment params and directory according to the test config

    experiment_params: ExperimentParams
    experiment_dir: Path
    experiment_date = config.experiment_date
    experiment_params, experiment_dir = find_experiment_config(
        experiments_dir=experiments_dir,
        experiment_date=experiment_date,
        test_config=test_experiment_config,
    )
    logging.debug("Experiment config found at %s", experiment_dir)
    if experiment_params.dataset_params.dataset_name == DatasetName.MNLI:
        dataset_params: MNLIParams = experiment_params.dataset_params
        dataset_params.combination_type = MNLICombinationType.CONCAT

    # Load the experiment type
    experiment_type: ExperimentType = ExperimentType(experiment_params.experiment_type)

    # Get the encoder weights path
    weights_dir = experiment_dir / "encoder_weights"
    if weights_dir is not None:
        if load_from_epoch is None:
            encoder_weights_path = sorted(
                weights_dir.glob("model_*.pt"), key=lambda x: int(x.stem.split("_")[-1])
            )[-1]
        else:
            encoder_weights_path = weights_dir / f"model_{load_from_epoch}.pt"

    # Create the test types based on the experiment type
    if experiment_type == ExperimentType.UTILITY:
        utility_privacy_tester = UtilityPrivacyTester(
            experiment_params=experiment_params,
            test_params=test_params,
            encoder_weights_path=encoder_weights_path,
        )
        utility_privacy_tester.run_all_tests()
    elif experiment_type == ExperimentType.COMPRESSION:
        compression_tester = CompressionTester(
            experiment_params=experiment_params,
            test_params=test_params,
            encoder_weights_path=encoder_weights_path,
        )
        compression_tester.run_all_tests()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
