from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import logging

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch

from src.utils.config import (
    ExperimentParams,
    EncoderParams,
    TestType,
    DatasetName,
)

from src.data.utils import TexShapeDataset
from src.utils.testing import TestClass
from src.utils.general import get_roc_auc
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model


@dataclass
class TestParams:
    """Test params dataclass."""

    max_epoch: int = -1
    batch_size: int = 2048
    device_idx: int = 3


def calculate_dimensions(experiment_params: ExperimentParams, utility: bool):
    """Calculate the dimensions for the model."""
    dataset_params = experiment_params.dataset_params
    dataset_name = dataset_params.dataset_name
    encoder_params: EncoderParams = experiment_params.encoder_params

    original_in_dim: int = 768 * 2 if dataset_name == DatasetName.MNLI else 768
    encoder_out_dim: int = encoder_params.encoder_model_params["out_dim"]
    texshape_in_dim: int = (
        encoder_out_dim * 2 if dataset_name == DatasetName.MNLI else encoder_out_dim
    )

    out_dim: int = 3 if dataset_name == DatasetName.MNLI and utility else 2
    return original_in_dim, texshape_in_dim, out_dim


def load_test_model(
    test_type: str, experiment_params: ExperimentParams = None, utility=True
) -> SimpleClassifier:
    """Load the test model."""
    original_in_dim, texshape_in_dim, out_dim = calculate_dimensions(
        experiment_params, utility
    )

    if test_type == TestType.RANDOM:
        model = SimpleClassifier(
            in_dim=texshape_in_dim, hidden_dims=[64], out_dim=out_dim
        )
    elif test_type == TestType.ORIGINAL:
        model = SimpleClassifier(
            in_dim=original_in_dim, hidden_dims=[64], out_dim=out_dim
        )
    elif test_type == TestType.TEXSHAPE:
        model = SimpleClassifier(
            in_dim=texshape_in_dim, hidden_dims=[64], out_dim=out_dim
        )
    elif test_type == TestType.NOISE:
        model = SimpleClassifier(
            in_dim=original_in_dim, hidden_dims=[64], out_dim=out_dim
        )
    elif test_type == TestType.QUANTIZATION:
        model = SimpleClassifier(
            in_dim=original_in_dim, hidden_dims=[64], out_dim=out_dim
        )
    else:
        raise ValueError(
            f"Test type {test_type} not supported. Supported types are {TestType}"
        )

    model.eval()
    return model


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
    model_name, model_params, weights_dir: Path = None, load_from_epoch=None
) -> Encoder:
    """Load and configure the encoder model."""
    logging.debug("Loading and configuring the encoder model")
    encoder_model = create_encoder_model(
        model_name=model_name, model_params=model_params
    )
    if weights_dir is not None:
        if load_from_epoch is None:
            encoder_weights_path = sorted(
                weights_dir.glob("model_*.pt"), key=lambda x: int(x.stem.split("_")[-1])
            )[-1]
        else:
            encoder_weights_path = weights_dir / f"model_{load_from_epoch}.pt"

        logging.debug("Encoder model weights path: %s", encoder_weights_path)
        encoder_model.load_state_dict(torch.load(encoder_weights_path))
    return encoder_model


def process_mnli_embeddings(
    encoder_model: Encoder,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Process the embeddings for the MNLI dataset through the encoder model."""
    premise_embeddings = embeddings[:, :768]
    hypothesis_embeddings = embeddings[:, 768:]
    encoded_premise_embeddings = encoder_model(premise_embeddings)
    encoded_hypothesis_embeddings = encoder_model(hypothesis_embeddings)
    # pylint: disable=no-member
    embeddings = torch.cat(
        (encoded_premise_embeddings, encoded_hypothesis_embeddings), dim=-1
    )
    return embeddings.detach()


def process_embeddings(
    encoder_model: Encoder,
    train_embeddings: torch.Tensor,
    validation_embeddings: torch.Tensor,
    device: torch.device,
    dataset_name: DatasetName,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process the embeddings through the encoder model."""
    encoder_model.eval()
    encoder_model.to(device)

    train_embeddings = train_embeddings.to(device)
    validation_embeddings = validation_embeddings.to(device)

    if dataset_name == DatasetName.MNLI:
        train_embeddings = process_mnli_embeddings(encoder_model, train_embeddings)
        validation_embeddings = process_mnli_embeddings(
            encoder_model, validation_embeddings
        )
    else:
        train_embeddings = encoder_model(train_embeddings)
        validation_embeddings = encoder_model(validation_embeddings)

    return train_embeddings.detach().cpu(), validation_embeddings.detach().cpu()


def train_model(
    pl_model: TestClass,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    max_epochs: int,
    device_idx: int,
) -> pl.Trainer:
    """
    Model training function.
    """
    logging.info("Training model")
    # scheduler = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    scheduler = pl.callbacks.EarlyStopping(
        monitor="val_acc", patience=20, mode="max", verbose=False
    )
    if device_idx == -1:
        accelerator = "cpu"
        trainer = pl.Trainer(
            accelerator=accelerator,
            callbacks=[scheduler],
            max_epochs=max_epochs,
        )
    else:
        accelerator = "gpu"
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=[device_idx],
            callbacks=[scheduler],
            max_epochs=max_epochs,
        )
    trainer.fit(
        model=pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )

    return trainer


def calculate_roc_and_auc(
    classifier_model: SimpleClassifier,
    validation_dataloader: DataLoader,
    device_idx: int,
):
    roc, auc = get_roc_auc(
        classifier_model, validation_dataloader, torch.device(f"cuda:{device_idx}")
    )
    roc = roc[:2]
    return roc, auc


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


def get_num_classes(experiment_params):
    dataset_params = experiment_params.dataset_params
    dataset_name = dataset_params.dataset_name
    if dataset_name == DatasetName.MNLI:
        num_class1 = 3
        num_class2 = 2
    else:
        num_class1 = 2
        num_class2 = 2
    return num_class1, num_class2
