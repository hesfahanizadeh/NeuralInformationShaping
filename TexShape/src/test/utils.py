"""Test utils."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import logging

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import numpy as np

from src.utils.config import (
    ExperimentParams,
    EncoderParams,
    TestType,
    DatasetName,
    ExperimentType,
)

from src.data.utils import TexShapeDataset
from src.utils.testing import TestClass
from src.utils.general import get_roc_auc
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model


@dataclass
class TestStats:
    """Test stats dataclass."""

    validation_roc: np.ndarray
    validation_auc_score: float
    validation_acc: float
    train_acc: float


@dataclass
class TestParams:
    """Test params dataclass."""

    max_epoch: int = -1
    batch_size: int = 2048
    device_idx: int = 3


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

    # # Create the test types based on the experiment type
    # test_types: List[ExperimentType] = create_test_types(experiment_type)

    # # Get num classes
    # num_class1, num_class2 = get_num_classes(experiment_params)

    # #
    # utility_rocs: Dict[Union[TestType, str], np.array] = {}
    # privacy_rocs = {}

    # utility_aucs = {}
    # privacy_aucs = {}

    # utility_accs = {}
    # privacy_accs = {}

    # for test_type in test_types:
    #     if test_type == TestType.NOISE:
    #         test_type: TestType
    #         noise_std_values = [0.1, 0.25, 0.5]

    #         for noise_std in noise_std_values:
    #             logging.info("Testing model for test type: %s", test_type)

    #             train_dataset, validation_dataset = load_train_dataset(
    #                 dataset_params=experiment_params.dataset_params, device=device
    #             )
    #             train_dataset.add_noise_embedding(noise_std)
    #             logging.info(
    #                 "Noise added to the training dataset. Noise std: %s", noise_std
    #             )

    #             (
    #                 train_utility_dataloader,
    #                 train_private_dataloader,
    #                 validation_utility_dataloader,
    #                 validation_private_dataloader,
    #             ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

    #             logging.info("Training the utility model")
    #             # Train the utility model
    #             classifier_model: SimpleClassifier = load_test_model(
    #                 test_type=test_type,
    #                 experiment_params=experiment_params,
    #                 utility=True,
    #             )
    #             # Train classifier
    #             pl_model = TestClass(model=classifier_model, num_class=num_class1)
    #             trainer = train_model(
    #                 pl_model=pl_model,
    #                 train_dataloader=train_utility_dataloader,
    #                 validation_dataloader=validation_utility_dataloader,
    #                 max_epochs=utility_max_epoch,
    #                 device_idx=device_idx,
    #             )
    #             roc, auc = calculate_roc_and_auc(
    #                 classifier_model=classifier_model,
    #                 validation_dataloader=validation_utility_dataloader,
    #                 device_idx=device_idx,
    #             )
    #             utility_rocs.update({f"Noise: {noise_std}": roc})
    #             utility_aucs.update({f"Noise: {noise_std}": auc})
    #             utility_accs.update({f"Noise: {noise_std}": (train_acc, val_acc)})

    #             logging.info("Training the privacy model")
    #             # Train the privacy model
    #             classifier_model: SimpleClassifier = load_test_model(
    #                 test_type=test_type,
    #                 experiment_params=experiment_params,
    #                 utility=False,
    #             )
    #             # Train classifier
    #             pl_model = TestClass(model=classifier_model, num_class=num_class2)
    #             tainer = train_model(
    #                 pl_model=pl_model,
    #                 train_dataloader=train_private_dataloader,
    #                 validation_dataloader=validation_private_dataloader,
    #                 max_epochs=privacy_max_epoch,
    #                 device_idx=device_idx,
    #             )
    #             roc, auc = calculate_roc_and_auc(
    #                 classifier_model, validation_private_dataloader, device_idx
    #             )
    #             privacy_rocs.update({f"Noise: {noise_std}": roc})
    #             privacy_aucs.update({f"Noise: {noise_std}": auc})
    #             privacy_accs.update({f"Noise: {noise_std}": (train_acc, val_acc)})

    # elif test_type == TestType.TEXSHAPE:
    #     logging.info("Testing model for test type: %s", test_type)

    #     train_dataset, validation_dataset = load_train_dataset(
    #         dataset_params=experiment_params.dataset_params, device=device
    #     )

    #     # Load the encoder model
    #     encoder_model_weights_dir = experiment_dir / "encoder_weights"
    #     encoder_model: Encoder = load_and_configure_encoder(
    #         model_name=experiment_params.encoder_params.encoder_model_name,
    #         model_params=experiment_params.encoder_params.encoder_model_params,
    #         weights_dir=encoder_model_weights_dir,
    #         load_from_epoch=load_from_epoch,
    #     )

    #     # Pass the embeddings through the encoder model
    #     train_dataset_embeddings, validation_dataset_embeddings = (
    #         process_embeddings(
    #             encoder_model=encoder_model,
    #             train_embeddings=train_dataset.embeddings,
    #             validation_embeddings=validation_dataset.embeddings,
    #             device=device,
    #             dataset_name=experiment_params.dataset_params.dataset_name,
    #         )
    #     )
    #     train_dataset.embeddings = train_dataset_embeddings
    #     validation_dataset.embeddings = validation_dataset_embeddings

    #     (
    #         train_utility_dataloader,
    #         train_private_dataloader,
    #         validation_utility_dataloader,
    #         validation_private_dataloader,
    #     ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

    #     logging.info("Training the utility model")
    #     # Train the utility model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=True
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class1)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )
    #     utility_aucs.update({test_type: auc})
    #     utility_rocs.update({test_type: roc})
    #     utility_accs.update({test_type: (train_acc, val_acc)})

    #     logging.info("Training the privacy model")
    #     # Train the privacy model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=False
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class2)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )
    #     privacy_aucs.update({test_type: auc})
    #     privacy_rocs.update({test_type: roc})
    #     privacy_accs.update({test_type: (train_acc, val_acc)})

    # elif test_type == TestType.RANDOM:
    #     logging.info("Testing model for test type: %s", test_type)

    #     train_dataset, validation_dataset = load_train_dataset(
    #         dataset_params=experiment_params.dataset_params, device=device
    #     )

    #     encoder_model: Encoder = load_and_configure_encoder(
    #         model_name=experiment_params.encoder_params.encoder_model_name,
    #         model_params=experiment_params.encoder_params.encoder_model_params,
    #     )

    #     # Pass the embeddings through the encoder model
    #     train_dataset_embeddings, validation_dataset_embeddings = (
    #         process_embeddings(
    #             encoder_model=encoder_model,
    #             train_embeddings=train_dataset.embeddings,
    #             validation_embeddings=validation_dataset.embeddings,
    #             device=device,
    #             dataset_name=experiment_params.dataset_params.dataset_name,
    #         )
    #     )
    #     train_dataset.embeddings = train_dataset_embeddings
    #     validation_dataset.embeddings = validation_dataset_embeddings

    #     (
    #         train_utility_dataloader,
    #         train_private_dataloader,
    #         validation_utility_dataloader,
    #         validation_private_dataloader,
    #     ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

    #     logging.info("Training the utility model")
    #     # Train the utility model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=True
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class1)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )

    #     utility_aucs.update({test_type: auc})
    #     utility_rocs.update({test_type: roc})
    #     utility_accs.update({test_type: (train_acc, val_acc)})

    #     logging.info("Training the privacy model")
    #     # Train the privacy model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=True
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class2)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )
    #     privacy_aucs.update({test_type: auc})
    #     privacy_rocs.update({test_type: roc})
    #     privacy_accs.update({test_type: (train_acc, val_acc)})

    # elif test_type == TestType.ORIGINAL:
    #     logging.info("Testing model for test type: %s", test_type)

    #     train_dataset, validation_dataset = load_train_dataset(
    #         dataset_params=experiment_params.dataset_params, device=device
    #     )
    #     (
    #         train_utility_dataloader,
    #         train_private_dataloader,
    #         validation_utility_dataloader,
    #         validation_private_dataloader,
    #     ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

    #     logging.info("Training the utility model")
    #     # Train the utility model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=True
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class1)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )

    #     utility_aucs.update({test_type: auc})
    #     utility_rocs.update({test_type: roc})
    #     utility_accs.update({test_type: (train_acc, val_acc)})

    #     logging.info("Training the privacy model")
    #     # Train the privacy model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=False
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class2)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )

    #     privacy_aucs.update({test_type: auc})
    #     privacy_rocs.update({test_type: roc})
    #     privacy_accs.update({test_type: (train_acc, val_acc)})

    # elif test_type == TestType.QUANTIZATION:
    #     logging.info("Testing model for test type: %s", test_type)

    #     train_dataset, validation_dataset = load_train_dataset(
    #         dataset_params=experiment_params.dataset_params, device=device
    #     )

    #     train_dataset.embeddings = quantize_embeddings(train_dataset.embeddings)
    #     validation_dataset.embeddings = quantize_embeddings(
    #         validation_dataset.embeddings
    #     )

    #     (
    #         train_utility_dataloader,
    #         train_private_dataloader,
    #         validation_utility_dataloader,
    #         validation_private_dataloader,
    #     ) = create_dataloaders(train_dataset, validation_dataset, batch_size)

    #     logging.info("Training the utility model")
    #     # Train the utility model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=True
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class1)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )

    #     utility_aucs.update({test_type: auc})
    #     utility_rocs.update({test_type: roc})
    #     utility_accs.update({test_type: (train_acc, val_acc)})

    #     logging.info("Training the privacy model")
    #     # Train the privacy model
    #     classifier_model: SimpleClassifier = load_test_model(
    #         test_type=test_type, experiment_params=experiment_params, utility=False
    #     )
    #     # Train classifier
    #     pl_model = TestClass(model=classifier_model, num_class=num_class2)
    #     train_acc, val_acc, roc, auc = train_model(
    #         pl_model=pl_model,
    #         train_dataloader=train_utility_dataloader,
    #         validation_dataloader=validation_utility_dataloader,
    #         max_epochs=utility_max_epoch,
    #         device_idx=device_idx,
    #     )
    #     privacy_aucs.update({test_type: auc})
    #     privacy_rocs.update({test_type: roc})
    #     privacy_accs.update({test_type: (train_acc, val_acc)})

    # names = [
    #     key.value.capitalize() if isinstance(key, TestType) else key.capitalize()
    #     for key in utility_rocs
    # ]

    # # Plot the rocs
    # plot_accs(
    #     accs=list(utility_rocs.values()),
    #     names=names,
    #     save_loc=experiment_dir / "utility_rocs.png",
    #     legend=True,
    # )

    # plot_accs(
    #     accs=list(privacy_rocs.values()),
    #     names=names,
    #     save_loc=experiment_dir / "privacy_rocs.png",
    #     legend=True,
    # )

    # raise NotImplementedError("Saving results not implemented.")


# for test_type, stats in self.test_type_stats.items():
#     first_dataset_stats: TestStats
#     second_dataset_stats: TestStats
#     first_dataset_stats, second_dataset_stats = stats

#     logging.info("Saving results for test type: %s", test_type)

# # Plot the rocs
# plot_accs(
#     accs=first_dataset_stats.validation_roc,
#     names=names,
#     save_loc=experiment_dir / "utility_rocs.png",
#     legend=True,
# )

# plot_accs(
#     accs=list(privacy_rocs.values()),
#     names=names,
#     save_loc=experiment_dir / "privacy_rocs.png",
#     legend=True,
# )

    def quantize_embeddings(
        self, embeddings: torch.Tensor, num_bits: int = 8
    ) -> torch.Tensor:
        """Quantize the embeddings."""
        # Shift the range of the data from [min, max] to [0, max - min]
        embeddings_min = embeddings.min()
        embeddings_max = embeddings.max()
        max_value = max(abs(embeddings_min), abs(embeddings_max))
        scale = max_value / 128

        # pylint: disable=no-member
        # Make the embeddings between
        quantized_embeddings = torch.quantize_per_tensor(
            embeddings, scale=scale, zero_point=0, dtype=torch.qint8
        )

        quantized_embeddings = torch.dequantize(quantized_embeddings)
        return embeddings