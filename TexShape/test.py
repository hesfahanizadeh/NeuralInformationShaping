"""
Test the models on the test dataset.
Author: H. Kaan Kale
Email: hkaankale1@gmail.com
"""

from pathlib import Path
from typing import Tuple, Union, List
import logging

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils.general import (
    set_seed,
    configure_torch_backend,
)
from src.utils.config import (
    ExperimentParams,
    ExperimentType,
    TestType,
    load_experiment_params,
)
from src.data.utils import load_experiment_dataset, TexShapeDataset
from src.utils.testing import TestClass
from src.utils.general import get_roc_auc
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model
from src.visualization.visualize import plot_accs


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

    raise ValueError(f"Experiment with params {config} not found in {experiments_dir}")


def create_test_types(experiment_type: ExperimentType) -> List[TestType]:
    """
    Create the experiment types based on the experiment type.

    :params experiment_type: str, the experiment type.
    :returns: List[TestTypes], the experiment types.
    """
    test_types = [
        TestType.RANDOM,
        TestType.ORIGINAL,
        TestType.TEXSHAPE,
    ]

    # Check if experiment type contains privacy
    if experiment_type == ExperimentType.UTILITY_PRIVACY:
        test_types.append(TestType.NOISE)
    elif experiment_type == ExperimentType.UTILITY:
        pass
    elif experiment_type == ExperimentType.COMPRESSION:
        test_types.append(TestType.QUANTIZATION)
    elif experiment_type == ExperimentType.COMPRESSION_PRIVACY:
        test_types.append(TestType.QUANTIZATION)
        test_types.append(TestType.NOISE)
    return test_types


def load_test_model(test_type: str) -> SimpleClassifier:
    """TODO: Fix this function."""

    if test_type == TestType.RANDOM:
        model = SimpleClassifier(in_dim=64, hidden_dims=[32], out_dim=2)
    elif test_type == TestType.ORIGINAL:
        model = SimpleClassifier(in_dim=768, hidden_dims=[32], out_dim=2)
    elif test_type == TestType.TEXSHAPE:
        model = SimpleClassifier(in_dim=64, hidden_dims=[32], out_dim=2)
    elif test_type == TestType.NOISE:
        model = SimpleClassifier(in_dim=768, hidden_dims=[32], out_dim=2)
    elif test_type == TestType.QUANTIZATION:
        model = SimpleClassifier(in_dim=768, hidden_dims=[32], out_dim=2)
    else:
        raise ValueError(
            f"Test type {test_type} not supported. Supported types are {TestType}"
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


def load_and_configure_encoder(
    model_name, model_params, device, weights_dir: Path = None
) -> Encoder:
    """Load and configure the encoder model."""
    logging.debug("Loading and configuring the encoder model")
    encoder_model = create_encoder_model(
        model_name=model_name, model_params=model_params
    )
    if weights_dir is not None:
        encoder_weights_path = sorted(
            weights_dir.glob("model_*.pt"), key=lambda x: int(x.stem.split("_")[-1])
        )[-1]
        logging.debug("Encoder model weights path: %s", encoder_weights_path)
        encoder_model.load_state_dict(torch.load(encoder_weights_path))

    encoder_model.eval()
    encoder_model.to(device)
    return encoder_model


def process_embeddings(
    encoder_model: Encoder,
    train_embeddings: torch.Tensor,
    validation_embeddings: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process the embeddings through the encoder model."""
    encoder_model.eval()
    encoder_model.to(device)
    train_embeddings = encoder_model(train_embeddings.to(device))
    validation_embeddings = encoder_model(validation_embeddings.to(device))
    return train_embeddings.detach(), validation_embeddings.detach()


def train_model(
    test_type, train_dataloader, validation_dataloader, max_epochs, device_idx
):
    """
    Model training function.
    """
    logging.info("Training model")
    classifier_model: SimpleClassifier

    classifier_model = load_test_model(test_type=test_type)

    # Train classifier
    pl_model = TestClass(model=classifier_model)
    # scheduler = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    scheduler = pl.callbacks.EarlyStopping(
        monitor="val_acc", patience=20, mode="max", verbose=False
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[device_idx],
        callbacks=[scheduler],
        max_epochs=max_epochs,
    )
    trainer.fit(
        model=pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )
    train_acc = trainer.callback_metrics.get("train_acc", 0).cpu().item()
    val_acc = trainer.callback_metrics.get("val_acc", 0).cpu().item()
    logging.info("Train Accuracy: %s", train_acc)
    logging.info("Validation Accuracy: %s", val_acc)

    roc, auc = get_roc_auc(
        classifier_model, validation_dataloader, torch.device(f"cuda:{device_idx}")
    )
    roc = roc[:2]
    return train_acc, val_acc, roc, auc


def quantize_embeddings(embeddings: torch.Tensor, num_bits=4) -> torch.Tensor:
    """Quantize the embeddings"""
    # Find the min and max values in the entire dataset
    min_value = embeddings.min()
    max_value = embeddings.max()

    upper_bound = 2**num_bits - 1

    if num_bits == 4:
        quantize_dtype = torch.int8
    elif num_bits == 8:
        quantize_dtype = torch.int16
    elif num_bits == 16:
        quantize_dtype = torch.int32

    # Normalize to [-1, 1]
    normalized_embeddings = 2 * (embeddings - min_value) / (max_value - min_value) - 1

    # Scale and shift to [0, upper_bound]
    scaled_embeddings = (normalized_embeddings + 1) / 2 * upper_bound

    # Quantize to 4 bits
    quantized_embeddings = torch.round(scaled_embeddings).to(quantize_dtype)  # pylint: disable=no-member

    # Dequantize
    dequantized_embeddings = quantized_embeddings.float() / upper_bound * 2 - 1

    # Denormalize
    denormalized_embeddings: torch.Tensor = (dequantized_embeddings + 1) / 2 * (
        max_value - min_value
    ) + min_value

    return denormalized_embeddings.detach()


@hydra.main(config_path="configs", config_name="test_config", version_base="1.2")
def main(config: DictConfig) -> None:
    """Main function to test the models on the test dataset."""
    # Configure the logger
    logging.basicConfig(level=logging.DEBUG)

    # Number of epochs for training the test classifiers
    utility_max_epoch: int = -1
    privacy_max_epoch: int = -1

    # Set the seed and configure the torch backend
    seed: int = 42
    set_seed(seed)
    configure_torch_backend()

    # Test config parameters
    batch_size: int = config.batch_size
    device_idx: int = config.device_idx
    experiments_dir: str = config.experiments_dir
    test_experiment_config = config.experiment
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    # Load experiment params and directory according to the test config

    experiment_params: ExperimentParams
    experiment_dir: Path
    experiment_params, experiment_dir = find_experiment_config(
        experiments_dir=experiments_dir, test_config=test_experiment_config
    )
    logging.debug("Experiment config found at %s", experiment_dir)

    # Load the experiment type
    experiment_type: ExperimentType = ExperimentType(experiment_params.experiment_type)

    # Create the test types based on the experiment type
    test_types: List[ExperimentType] = create_test_types(experiment_type)

    #
    utility_rocs = {}
    privacy_rocs = {}

    utility_aucs = {}
    privacy_aucs = {}

    utility_accs = {}
    privacy_accs = {}

    for test_type in test_types:
        if test_type == TestType.NOISE:
            test_type: TestType
            noise_std_values = [0.1, 0.25, 0.5]

            for noise_std in noise_std_values:
                logging.info("Testing model for test type: %s", test_type)

                train_dataset, validation_dataset = load_experiment_dataset(
                    dataset_params=experiment_params.dataset_params, device=device
                )
                train_dataset.add_noise_embedding(noise_std)
                logging.info(
                    "Noise added to the training dataset. Noise std: %s", noise_std
                )

                (
                    train_utility_dataloader,
                    train_private_dataloader,
                    validation_utility_dataloader,
                    validation_private_dataloader,
                ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

                logging.info("Training the utility model")
                # Train the utility model
                train_acc, val_acc, roc, auc = train_model(
                    test_type,
                    train_utility_dataloader,
                    validation_utility_dataloader,
                    utility_max_epoch,
                    device_idx,
                )
                utility_rocs.update({f"Noise: {noise_std}": roc})
                utility_aucs.update({f"Noise: {noise_std}": auc})
                utility_accs.update({f"Noise: {noise_std}": (train_acc, val_acc)})

                logging.info("Training the privacy model")
                # Train the privacy model
                train_acc, val_acc, roc, auc = train_model(
                    test_type,
                    train_private_dataloader,
                    validation_private_dataloader,
                    privacy_max_epoch,
                    device_idx,
                )
                privacy_rocs.update({f"Noise: {noise_std}": roc})
                privacy_aucs.update({f"Noise: {noise_std}": auc})
                privacy_accs.update({f"Noise: {noise_std}": (train_acc, val_acc)})

        elif test_type == TestType.TEXSHAPE:
            logging.info("Testing model for test type: %s", test_type)

            train_dataset, validation_dataset = load_experiment_dataset(
                dataset_params=experiment_params.dataset_params, device=device
            )

            # Load the encoder model
            encoder_model_weights_dir = experiment_dir / "encoder_weights"
            encoder_model: Encoder = load_and_configure_encoder(
                model_name=experiment_params.encoder_params.encoder_model_name,
                model_params=experiment_params.encoder_params.encoder_model_params,
                device=device,
                weights_dir=encoder_model_weights_dir,
            )

            # Pass the embeddings through the encoder model
            train_dataset_embeddings, validation_dataset_embeddings = (
                process_embeddings(
                    encoder_model=encoder_model,
                    train_embeddings=train_dataset.embeddings,
                    validation_embeddings=validation_dataset.embeddings,
                    device=device,
                )
            )
            train_dataset.embeddings = train_dataset_embeddings
            validation_dataset.embeddings = validation_dataset_embeddings

            (
                train_utility_dataloader,
                train_private_dataloader,
                validation_utility_dataloader,
                validation_private_dataloader,
            ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

            logging.info("Training the utility model")
            # Train the utility model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_utility_dataloader,
                validation_utility_dataloader,
                utility_max_epoch,
                device_idx,
            )
            utility_aucs.update({test_type: auc})
            utility_rocs.update({test_type: roc})
            utility_accs.update({test_type: (train_acc, val_acc)})

            logging.info("Training the privacy model")
            # Train the privacy model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_private_dataloader,
                validation_private_dataloader,
                privacy_max_epoch,
                device_idx,
            )

            privacy_aucs.update({test_type: auc})
            privacy_rocs.update({test_type: roc})
            privacy_accs.update({test_type: (train_acc, val_acc)})

        elif test_type == TestType.RANDOM:
            logging.info("Testing model for test type: %s", test_type)

            train_dataset, validation_dataset = load_experiment_dataset(
                dataset_params=experiment_params.dataset_params, device=device
            )

            encoder_model: Encoder = load_and_configure_encoder(
                model_name=experiment_params.encoder_params.encoder_model_name,
                model_params=experiment_params.encoder_params.encoder_model_params,
                device=device,
            )

            # Pass the embeddings through the encoder model
            train_dataset_embeddings, validation_dataset_embeddings = (
                process_embeddings(
                    encoder_model=encoder_model,
                    train_embeddings=train_dataset.embeddings,
                    validation_embeddings=validation_dataset.embeddings,
                    device=device,
                )
            )
            train_dataset.embeddings = train_dataset_embeddings
            validation_dataset.embeddings = validation_dataset_embeddings

            (
                train_utility_dataloader,
                train_private_dataloader,
                validation_utility_dataloader,
                validation_private_dataloader,
            ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

            logging.info("Training the utility model")
            # Train the utility model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_utility_dataloader,
                validation_utility_dataloader,
                utility_max_epoch,
                device_idx,
            )

            utility_aucs.update({test_type: auc})
            utility_rocs.update({test_type: roc})
            utility_accs.update({test_type: (train_acc, val_acc)})

            logging.info("Training the privacy model")
            # Train the privacy model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_private_dataloader,
                validation_private_dataloader,
                privacy_max_epoch,
                device_idx,
            )
            privacy_aucs.update({test_type: auc})
            privacy_rocs.update({test_type: roc})
            privacy_accs.update({test_type: (train_acc, val_acc)})

        elif test_type == TestType.ORIGINAL:
            logging.info("Testing model for test type: %s", test_type)

            train_dataset, validation_dataset = load_experiment_dataset(
                dataset_params=experiment_params.dataset_params, device=device
            )
            (
                train_utility_dataloader,
                train_private_dataloader,
                validation_utility_dataloader,
                validation_private_dataloader,
            ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

            logging.info("Training the utility model")
            # Train the utility model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_utility_dataloader,
                validation_utility_dataloader,
                utility_max_epoch,
                device_idx,
            )

            utility_aucs.update({test_type: auc})
            utility_rocs.update({test_type: roc})
            utility_accs.update({test_type: (train_acc, val_acc)})

            logging.info("Training the privacy model")
            # Train the privacy model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_private_dataloader,
                validation_private_dataloader,
                privacy_max_epoch,
                device_idx,
            )

            privacy_aucs.update({test_type: auc})
            privacy_rocs.update({test_type: roc})
            privacy_accs.update({test_type: (train_acc, val_acc)})

        elif test_type == TestType.QUANTIZATION:
            logging.info("Testing model for test type: %s", test_type)

            train_dataset, validation_dataset = load_experiment_dataset(
                dataset_params=experiment_params.dataset_params, device=device
            )

            train_dataset.embeddings = quantize_embeddings(train_dataset.embeddings)
            validation_dataset.embeddings = quantize_embeddings(
                validation_dataset.embeddings
            )

            (
                train_utility_dataloader,
                train_private_dataloader,
                validation_utility_dataloader,
                validation_private_dataloader,
            ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

            logging.info("Training the utility model")
            # Train the utility model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_utility_dataloader,
                validation_utility_dataloader,
                utility_max_epoch,
                device_idx,
            )

            utility_aucs.update({test_type: auc})
            utility_rocs.update({test_type: roc})
            utility_accs.update({test_type: (train_acc, val_acc)})

            logging.info("Training the privacy model")
            # Train the privacy model
            train_acc, val_acc, roc, auc = train_model(
                test_type,
                train_private_dataloader,
                validation_private_dataloader,
                privacy_max_epoch,
                device_idx,
            )

            privacy_aucs.update({test_type: auc})
            privacy_rocs.update({test_type: roc})
            privacy_accs.update({test_type: (train_acc, val_acc)})

    names = [
        key.value.capitalize() if isinstance(key, TestType) else key.capitalize()
        for key in utility_rocs
    ]

    # Plot the rocs

    plot_accs(
        accs=list(utility_rocs.values()),
        names=names,
        save_loc=experiment_dir / "utility_rocs.png",
        legend=True,
    )

    plot_accs(
        accs=list(privacy_rocs.values()),
        names=names,
        save_loc=experiment_dir / "privacy_rocs.png",
        legend=True,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
