"""
Test the models on the test dataset.
# TODO: Finish the test script and the documentation.
"""

from pathlib import Path
from typing import Tuple, Union, List
import logging
from enum import Enum

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils.general import (
    set_seed,
    configure_torch_backend,
    load_experiment_params,
    ExperimentParams,
)
from src.data.utils import load_experiment_dataset, load_dataset_params, TexShapeDataset
from src.utils.testing import TestClass
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model


class TestTypes(Enum):
    """Test types."""

    RANDOM = "RANDOM"
    ORIGINAL = "ORIGINAL"
    TEXSHAPE = "TEXSHAPE"
    NOISE = "NOISE"


class ExperimentTypes(Enum):
    """Experiment types."""

    UTILITY = "utility"
    UTILITY_PRIVACY = "utility+privacy"
    COMPRESSION = "compression"
    COMPRESSION_PRIVACY = "compression+privacy"


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
    elif test_config is not None:
        experiment_params, experiment_dir = find_config_in_experiments_by_config(
            config=test_config, experiments_dir=experiments_dir
        )
        return experiment_params, experiment_dir

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


def create_test_types(experiment_type: str) -> List[str]:
    """
    Create the experiment types based on the experiment type.

    :params experiment_type: str, the experiment type.
    :returns: List[str], the experiment types.
    """
    test_types = ["ORIGINAL"] # RANDOM ,, "TEXSHAPE"

    # Check if experiment type contains privacy
    if experiment_type == ExperimentTypes.UTILITY_PRIVACY.value:
        test_types.append("NOISE")
    elif experiment_type == ExperimentTypes.UTILITY.value:
        pass
    elif experiment_type == ExperimentTypes.COMPRESSION.value:
        pass
    elif experiment_type == ExperimentTypes.COMPRESSION_PRIVACY.value:
        test_types.append("NOISE")
    return test_types


def load_test_model(test_type: str) -> SimpleClassifier:
    """TODO: Fix this function."""

    if test_type == TestTypes.RANDOM.value:
        model = SimpleClassifier(in_dim=768, hidden_dims=[64], out_dim=2)
    elif test_type == TestTypes.ORIGINAL.value:
        model = SimpleClassifier(in_dim=768, hidden_dims=[64], out_dim=2)
    elif test_type == TestTypes.TEXSHAPE.value:
        model = SimpleClassifier(in_dim=64, hidden_dims=[64], out_dim=2)
    elif test_type == TestTypes.NOISE.value:
        model = SimpleClassifier(in_dim=768, hidden_dims=[64], out_dim=2)
    elif not hasattr(TestTypes, test_type):
        raise ValueError(
            f"Test type {test_type} not supported. Supported types are {TestTypes}"
        )

    model.eval()
    return model
    #  model.load_state_dict(torch.load(f"models/{test_type}.pth"))


def create_dataloaders(
    train_dataset: TexShapeDataset, validation_dataset: TexShapeDataset, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Load the utility and privacy dataloaders."""
    train_utility_dataset = TensorDataset(
        train_dataset.embeddings, train_dataset.label1
    )
    train_utility_dataloader = DataLoader(
        dataset=train_utility_dataset, batch_size=batch_size, shuffle=True
    )

    train_private_datset = TensorDataset(train_dataset.embeddings, train_dataset.label2)
    train_private_dataloader = DataLoader(
        dataset=train_private_datset, batch_size=batch_size, shuffle=True
    )

    validation_utility_dataset = TensorDataset(
        validation_dataset.embeddings, validation_dataset.label1
    )
    validation_utility_dataloader = DataLoader(
        dataset=validation_utility_dataset, batch_size=batch_size, shuffle=False
    )

    validation_private_dataset = TensorDataset(
        validation_dataset.embeddings, validation_dataset.label2
    )
    validation_private_dataloader = DataLoader(
        dataset=validation_private_dataset, batch_size=batch_size, shuffle=False
    )

    return (
        train_utility_dataloader,
        train_private_dataloader,
        validation_utility_dataloader,
        validation_private_dataloader,
    )

@hydra.main(config_path="configs", config_name="test_config", version_base="1.2")
def main(test_config: DictConfig) -> None:
    """Main function to test the models on the test dataset."""
    # Configure the logger
    logging.basicConfig(level=logging.DEBUG)

    seed: int = 42
    set_seed(seed)
    configure_torch_backend()

    # TODO: Make this a dataclass
    batch_size: int = test_config.batch_size
    device_idx: int = test_config.device_idx
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    # Load experiment params and directory according to the test config
    experiments_dir: str = test_config.experiments_dir
    test_experiment_config = test_config.experiment

    experiment_params: ExperimentParams
    experiment_dir: Path
    experiment_params, experiment_dir = find_experiment_config(
        experiments_dir=experiments_dir, test_config=test_experiment_config
    )

    logging.debug("Experiment config found at %s", experiment_dir)

    # Create experiment_params
    dataset_name = experiment_params.dataset_name
    dataset_params = load_dataset_params(dataset_name, test_experiment_config)

    experiment_type: str = experiment_params.experiment_type
    test_types = create_test_types(experiment_type)

    test_types = ["NOISE"]
    for test_type in test_types:
        logging.info("Testing model for test type: %s", test_type)
        utility_model: SimpleClassifier = load_test_model(test_type=test_type)
        privacy_model: SimpleClassifier = load_test_model(test_type=test_type)
        
        train_dataset, validation_dataset = load_experiment_dataset(
            dataset_params=dataset_params, device=device
        )

        if test_type == TestTypes.NOISE.value:
            noise_std = 0.2
            train_dataset.add_noise_embedding(noise_std)
            # TODO: Only add noise to the original training dataset
            # validation_dataset.add_noise_embedding(noise_std)

        elif test_type == TestTypes.TEXSHAPE.value:
            # Load the encoder model
            encoder_model: Encoder = create_encoder_model(
                model_name=experiment_params.encoder_params.encoder_model_name,
                model_params=experiment_params.encoder_params.encoder_model_params,
            )
            encoder_model_weights_dir = experiment_dir / "encoder_weights"

            # Load the last epoch encoder model weights
            encoder_model_weights = sorted(
                encoder_model_weights_dir.glob("model_*.pt"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )[-1]

            # Log the encoder model weights path as a debug message
            logging.debug("Encoder model weights path: %s", encoder_model_weights)

            # Load the encoder model weights
            encoder_model = torch.load(encoder_model_weights)
            encoder_model.eval()
            encoder_model.to(device)

            # Pass the embeddings through the encoder model
            train_dataset_embeddings = encoder_model(
                train_dataset.embeddings.to(device)
            )
            validation_dataset_embeddings = encoder_model(
                validation_dataset.embeddings.to(device)
            )

            train_dataset.embeddings = train_dataset_embeddings
            validation_dataset.embeddings = validation_dataset_embeddings

        elif test_type == TestTypes.RANDOM.value:
            encoder_model: Encoder = create_encoder_model(
                model_name=experiment_params.encoder_params.encoder_model_name,
                model_params=experiment_params.encoder_params.encoder_model_params,
            )

        (
            train_utility_dataloader,
            train_private_dataloader,
            validation_utility_dataloader,
            validation_private_dataloader,
        ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

        # Train classifier
        # Utility model
        utility_m = TestClass(model=utility_model)
        trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=[device_idx])
        trainer.fit(model=utility_m, train_dataloaders=train_utility_dataloader, val_dataloaders=validation_utility_dataloader)

        # Privacy model
        privacy_m = TestClass(model=privacy_model)
        trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=[device_idx])
        trainer.fit(model=privacy_m, train_dataloaders=train_private_dataloader, val_dataloaders=validation_private_dataloader)

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
