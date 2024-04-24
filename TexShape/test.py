from pathlib import Path
from typing import Tuple, Union, List
import logging

from torch.utils.data import DataLoader
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils.general import (
    set_seed,
    configure_torch_backend,
    load_experiment_params,
    ExperimentParams,
)
from src.data.utils import load_experiment_dataset, load_dataset_params
from src.utils.testing import TestClass
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model

TEST_TYPES = ["RANDOM", "ORIGINAL", "TEXSHAPE", "NOISE"]


def find_experiment_config(
    *,
    experiments_dir: Union[Path, str],
    experiment_date: str = None,
    test_config: DictConfig = None,
) -> Tuple[ExperimentParams, Path]:
    """
    Find the experiment config file in the experiments directory.
    If experiment_date is provided, load the experiment config file from the experiments directory.
    If experiment_date is not provided, find the experiment with the same params as the test_experiment_config.

    :params test_config: DictConfig, the test config file.
    :params experiments_dir: Path, the experiments directory path.
    :params experiment_date: str, the experiment date to load the experiment config file.
    :returns: ExperimentParams, the experiment params of the experiment config file. Path, Experiment directory path
    :raises: ValueError, if the experiment with the same params as the test_experiment_config is not found.
    """

    # If experiment_date is provided, load the experiment config file from the experiments directory
    if experiment_date is not None:
        # Find the experiment config file in the experiments directory
        experiment_params, experiment_dir = find_config_in_experiments_by_date(
            experiment_date=experiment_date, experiments_dir=experiments_dir
        )
        return experiment_params, experiment_dir

    # If experiment_date is not provided, find the experiment with the same params as the test_experiment_config
    elif test_config is not None:
        experiment_params, experiment_dir = find_config_in_experiments_by_config(
            config=test_config, experiments_dir=experiments_dir
        )
        return experiment_params, experiment_dir

    raise ValueError(
        f"Experiment with params {test_config} not found in {experiments_dir} and experiment_date {experiment_date}"
    )


def find_config_in_experiments_by_date(
    experiment_date: str, experiments_dir: Union[Path, str]
) -> Tuple[ExperimentParams, Path]:
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
    test_types = ["RANDOM", "ORIGINAL", "TEXSHAPE"]

    # Check if experiment type contains privacy
    if (
        experiment_type == "utility+privacy"
    ):  # choices=["utility", "utility+privacy", "compression", "compression+privacy"]experiment_type:
        test_types.append("NOISE")
    elif experiment_type == "utility":
        pass
    elif experiment_type == "compression":
        pass
    elif experiment_type == "compression+privacy":
        test_types.append("NOISE")
    return test_types


def load_test_model(test_type: str, dataset_name: str) -> SimpleClassifier:
    """TODO: Fix this function."""

    if test_type == "RANDOM":
        model = SimpleClassifier(in_dim=768, hidden_dims=[64], out_dim=2)
    elif test_type == "ORIGINAL":
        model = SimpleClassifier(in_dim=768, hidden_dims=[64], out_dim=2)
    elif test_type == "TEXSHAPE":
        model = SimpleClassifier(in_dim=64, hidden_dims=[64], out_dim=2)
    elif test_type == "NOISE":
        model = SimpleClassifier(in_dim=768, hidden_dims=[64], out_dim=2)

    elif test_type not in TEST_TYPES:
        raise ValueError(
            f"Test type {test_type} not supported. Supported types are {TEST_TYPES}"
        )

    model.eval()
    return model
    #  model.load_state_dict(torch.load(f"models/{test_type}.pth"))


@hydra.main(config_path="configs", config_name="test_config", version_base="1.2")
def main(test_config: DictConfig) -> None:
    # Configure the logger
    logging.basicConfig(level=logging.DEBUG)
    
    # TODO: Fix, not finished
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

    # Create experiment_params
    dataset_name = experiment_params.dataset_name
    dataset_params = load_dataset_params(dataset_name, test_experiment_config)

    experiment_type: str = experiment_params.experiment_type
    test_types = create_test_types(experiment_type)
    test_types = ["NOISE"]

    test_types = ["TEXSHAPE"]
    for test_type in test_types:
        model: SimpleClassifier = load_test_model(
            test_type=test_type, dataset_name=dataset_name
        )

        train_dataset, validation_dataset = load_experiment_dataset(
            dataset_params=dataset_params, device=device
        )

        if test_type == "NOISE":
            noise_std = 0.1
            train_dataset = train_dataset.add_noise_embedding(noise_std)
            validation_dataset = validation_dataset.add_noise_embedding(noise_std)

        elif test_type == "TEXSHAPE":
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
            logging.debug(f"Encoder model weights path: {encoder_model_weights}")
            
            # Load the encoder model weights
            encoder_model = torch.load(encoder_model_weights)
            encoder_model.eval()
            encoder_model.to(device)
            
            # Pass the embeddings through the encoder model
            train_dataset_embeddings = encoder_model(train_dataset.embeddings.to(device))
            validation_dataset_embeddings = encoder_model(validation_dataset.embeddings.to(device))
            
            train_dataset.embeddings = train_dataset_embeddings
            validation_dataset.embeddings = validation_dataset_embeddings
            
        # Create data loaders
        train_data_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_data_loader = DataLoader(
            dataset=validation_dataset, batch_size=batch_size, shuffle=False
        )

        # Define the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        tester = TestClass(
            model=model,
            train_loader=train_data_loader,
            val_loader=validation_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        epochs: int = 20

        for epoch in range(epochs):
            train_loss, train_accuracy = tester.train_one_epoch()
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}"
            )

            # Validation
            validation_loss, validation_accuracy = tester.test_function()
            print(
                f"Epoch: {epoch}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}"
            )


if __name__ == "__main__":
    main()
