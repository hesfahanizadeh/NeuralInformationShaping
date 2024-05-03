"""Compression test module."""

from pathlib import Path
from typing import Tuple, Dict
import logging
import pickle

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import numpy as np

from src.utils.config import ExperimentParams, EncoderParams, TestType, DatasetName
from src.utils.testing import TestClass
from src.utils.general import get_roc_auc
from src.data.utils import load_train_dataset, TexShapeDataset
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model
from src.test.utils import TestParams, TestStats


TEST_TYPES = [
    TestType.RANDOM,
    TestType.ORIGINAL,
    TestType.TEXSHAPE,
    # TestType.NOISE,
    TestType.QUANTIZATION,
]


class CompressionTester:
    """Utility privacy tester class."""

    def __init__(
        self,
        experiment_params: ExperimentParams,
        test_params: TestParams,
        encoder_weights_path: Path,
    ):
        self.test_params = test_params
        self.device = torch.device(
            f"cuda:{test_params.device_idx}"
            if torch.cuda.is_available() and test_params.device_idx != -1
            else "cpu"
        )
        self.experiment_params = experiment_params
        self.test_type_stats: Dict[TestType : Tuple[TestStats, TestStats]] = {}
        self.encoder_weights_path = encoder_weights_path

    def run_all_tests(self):
        """Run all tests."""
        for test_type in TEST_TYPES:
            self.run_test_type(test_type)

        self.save_results()

    def run_test_type(self, test_type: TestType):
        """Run the test for the given test type."""
        max_epoch = self.test_params.max_epoch
        logging.info("Testing model for test type: %s", test_type)
        (
            train_first_dataloader,
            validation_first_dataloader,
            train_second_dataloader,
            validation_second_dataloader,
        ) = self.create_dataloaders(test_type=test_type)

        if self.experiment_params.dataset_params.dataset_name == DatasetName.MNLI:
            return_roc_auc_first_dataset = False
        else:
            return_roc_auc_first_dataset = True

        first_dataset_stats = self.train_and_evaluate_model(
            test_type,
            train_first_dataloader,
            validation_first_dataloader,
            max_epoch,
            return_roc_auc=return_roc_auc_first_dataset,
        )

        second_dataset_stats = self.train_and_evaluate_model(
            test_type,
            train_second_dataloader,
            validation_second_dataloader,
            max_epoch,
            return_roc_auc=True,
        )

        self.test_type_stats[test_type] = (first_dataset_stats, second_dataset_stats)

    def train_and_evaluate_model(
        self,
        test_type: TestType,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        max_epochs: int,
        return_roc_auc: bool = True,
    ) -> TestStats:
        """Train and evaluate the model."""
        # Get the number of classes
        validation_dataset: TensorDataset = validation_dataloader.dataset
        unique_class_values: torch.Tensor = validation_dataset.tensors[1].unique()
        num_class = unique_class_values.shape[0]

        classifier_model: SimpleClassifier = self.load_test_model(
            test_type=test_type, out_dim=num_class
        )
        # Initiailize the Test Class
        pl_model = TestClass(model=classifier_model, num_class=num_class)
        trainer = self.train_model(
            pl_model=pl_model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            max_epochs=max_epochs,
        )
        if return_roc_auc:
            roc_auc_results = self.get_roc_auc(classifier_model, validation_dataloader)
        else:
            roc_auc_results = (np.array([0, 0]), 0)
        train_acc: float = trainer.callback_metrics.get("train_acc", 0).cpu().item()
        val_acc: float = trainer.callback_metrics.get("val_acc", 0).cpu().item()
        return TestStats(
            validation_roc=roc_auc_results[0],
            validation_auc_score=roc_auc_results[1],
            validation_acc=val_acc,
            train_acc=train_acc,
        )

    def train_model(
        self,
        pl_model: TestClass,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        max_epochs: int,
    ) -> pl.Trainer:
        """
        Model training function.
        """
        logging.debug("Training model")
        # scheduler = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        scheduler = pl.callbacks.EarlyStopping(
            monitor="val_acc", patience=20, mode="max", verbose=False
        )
        if self.device == "cpu":
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
                devices=[self.device.index],
                callbacks=[scheduler],
                max_epochs=max_epochs,
            )
        trainer.fit(
            model=pl_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=validation_dataloader,
        )
        return trainer

    def create_first_dataset(
        self, train_dataset: TexShapeDataset, validation_dataset: TexShapeDataset
    ):
        train_utility_dataset = TensorDataset(
            train_dataset.embeddings, train_dataset.label1
        )
        validation_utility_dataset = TensorDataset(
            validation_dataset.embeddings, validation_dataset.label1
        )
        return train_utility_dataset, validation_utility_dataset

    def create_second_dataset(
        self, train_dataset: TexShapeDataset, validation_dataset: TexShapeDataset
    ):
        train_private_datset = TensorDataset(
            train_dataset.embeddings, train_dataset.label2
        )
        validation_private_dataset = TensorDataset(
            validation_dataset.embeddings, validation_dataset.label2
        )
        return train_private_datset, validation_private_dataset

    def create_dataloaders(
        self,
        test_type: TestType,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load the utility and privacy dataloaders."""
        batch_size = self.test_params.batch_size
        train_dataset: TexShapeDataset
        validation_dataset: TexShapeDataset
        train_dataset, validation_dataset = load_train_dataset(
            dataset_params=self.experiment_params.dataset_params, device=self.device
        )
        train_dataset, validation_dataset = self.preprocess_data(
            train_dataset, validation_dataset, test_type
        )

        first_datasets = self.create_first_dataset(train_dataset, validation_dataset)
        train_first_dataloader = DataLoader(
            dataset=first_datasets[0], batch_size=batch_size, shuffle=True
        )
        validation_first_dataloader = DataLoader(
            dataset=first_datasets[1],
            batch_size=batch_size,
            shuffle=False,
        )
        second_datasets = self.create_second_dataset(train_dataset, validation_dataset)
        train_second_dataloader = DataLoader(
            dataset=second_datasets[0], batch_size=batch_size, shuffle=True
        )
        validation_second_dataloader = DataLoader(
            dataset=second_datasets[1],
            batch_size=batch_size,
            shuffle=False,
        )

        return (
            train_first_dataloader,
            validation_first_dataloader,
            train_second_dataloader,
            validation_second_dataloader,
        )

    def preprocess_data(
        self,
        train_dataset: TexShapeDataset,
        validation_dataset: TexShapeDataset,
        test_type: TestType,
    ):
        """Preprocess the data."""
        if test_type == TestType.RANDOM:
            train_dataset_embeddings, validation_dataset_embeddings = (
                self.encode_embeddings(test_type, train_dataset, validation_dataset)
            )
            train_dataset.embeddings = train_dataset_embeddings
            validation_dataset.embeddings = validation_dataset_embeddings
            return train_dataset, validation_dataset
        if test_type == TestType.ORIGINAL:
            return train_dataset, validation_dataset
        if test_type == TestType.TEXSHAPE:
            train_dataset_embeddings, validation_dataset_embeddings = (
                self.encode_embeddings(test_type, train_dataset, validation_dataset)
            )
            train_dataset.embeddings = train_dataset_embeddings
            validation_dataset.embeddings = validation_dataset_embeddings
            return train_dataset, validation_dataset
        if test_type == TestType.QUANTIZATION:
            train_dataset.embeddings = self.quantize_embeddings(
                train_dataset.embeddings
            )
            validation_dataset.embeddings = self.quantize_embeddings(
                validation_dataset.embeddings
            )
            return train_dataset, validation_dataset
        raise ValueError(f"Test type {test_type} not supported.")

    def calculate_dimensions(self):
        """Calculate the dimensions for the model."""
        dataset_params = self.experiment_params.dataset_params
        dataset_name = dataset_params.dataset_name
        encoder_params: EncoderParams = self.experiment_params.encoder_params

        original_in_dim: int = 768 * 2 if dataset_name == DatasetName.MNLI else 768
        encoder_out_dim: int = encoder_params.encoder_model_params["out_dim"]
        texshape_in_dim: int = (
            encoder_out_dim * 2 if dataset_name == DatasetName.MNLI else encoder_out_dim
        )

        return original_in_dim, texshape_in_dim

    def load_test_model(self, test_type: str, out_dim: int) -> SimpleClassifier:
        """Load the test model."""
        original_in_dim, texshape_in_dim = self.calculate_dimensions()

        if test_type == TestType.RANDOM:
            model = SimpleClassifier(
                in_dim=texshape_in_dim, hidden_dims=[128, 64], out_dim=out_dim
            )
        elif test_type == TestType.ORIGINAL:
            model = SimpleClassifier(
                in_dim=original_in_dim, hidden_dims=[128, 64], out_dim=out_dim
            )
        elif test_type == TestType.TEXSHAPE:
            model = SimpleClassifier(
                in_dim=texshape_in_dim, hidden_dims=[128, 64], out_dim=out_dim
            )
        elif test_type == TestType.QUANTIZATION:
            model = SimpleClassifier(
                in_dim=original_in_dim, hidden_dims=[128, 64], out_dim=out_dim
            )
        else:
            raise ValueError(
                f"Test type {test_type} not supported. Supported types are {TestType}"
            )

        model.eval()
        return model

    def quantize_embeddings(self, embeddings: torch.Tensor, num_bits=4) -> torch.Tensor:
        """Quantize the embeddings"""
        # Find the min and max values in the entire dataset
        min_value = embeddings.min()
        max_value = embeddings.max()

        upper_bound = 2**num_bits - 1

        if num_bits == 1:
            embeddings *= 8
            return embeddings.round().detach()
        elif num_bits == 4:
            quantize_dtype = torch.int8
        elif num_bits == 8:
            quantize_dtype = torch.int16
        elif num_bits == 16:
            quantize_dtype = torch.int32

        # Normalize to [-1, 1]
        normalized_embeddings = (
            2 * (embeddings - min_value) / (max_value - min_value) - 1
        )

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

    def encode_embeddings(
        self,
        test_type: TestType,
        train_dataset: TexShapeDataset,
        validation_dataset: TexShapeDataset,
    ):
        """Encode the embeddings."""
        encoder_weights_path = (
            self.encoder_weights_path if test_type == test_type.TEXSHAPE else None
        )
        # Load the encoder model
        encoder_model: Encoder = self.load_and_configure_encoder(
            model_name=self.experiment_params.encoder_params.encoder_model_name,
            model_params=self.experiment_params.encoder_params.encoder_model_params,
            encoder_weights_path=encoder_weights_path,
        )

        # Pass the embeddings through the encoder model
        train_dataset_embeddings, validation_dataset_embeddings = (
            self.process_embeddings(
                encoder_model=encoder_model,
                train_embeddings=train_dataset.embeddings,
                validation_embeddings=validation_dataset.embeddings,
                device=self.device,
            )
        )
        return train_dataset_embeddings, validation_dataset_embeddings

    def process_embeddings(
        self,
        encoder_model: Encoder,
        train_embeddings: torch.Tensor,
        validation_embeddings: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the embeddings through the encoder model."""
        encoder_model.eval()
        encoder_model.to(device)

        train_embeddings = train_embeddings.to(device)
        validation_embeddings = validation_embeddings.to(device)

        if self.experiment_params.dataset_params.dataset_name == DatasetName.MNLI:
            train_embeddings = self.process_mnli_embeddings(
                encoder_model, train_embeddings
            )
            validation_embeddings = self.process_mnli_embeddings(
                encoder_model, validation_embeddings
            )
        else:
            train_embeddings = encoder_model(train_embeddings)
            validation_embeddings = encoder_model(validation_embeddings)

        return train_embeddings.detach().cpu(), validation_embeddings.detach().cpu()

    def process_mnli_embeddings(
        self,
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

    def load_and_configure_encoder(
        self, model_name, model_params, encoder_weights_path: Path = None
    ) -> Encoder:
        """Load and configure the encoder model."""
        logging.debug("Loading and configuring the encoder model")
        encoder_model = create_encoder_model(
            model_name=model_name, model_params=model_params
        )
        if encoder_weights_path is not None:
            logging.debug("Encoder model weights path: %s", encoder_weights_path)
            encoder_model.load_state_dict(torch.load(encoder_weights_path))
        return encoder_model

    def get_roc_auc(self, classifier_model, validation_dataloader):
        """Get ROC and AUC for the given model."""
        roc, auc = get_roc_auc(classifier_model, validation_dataloader, self.device)
        roc = roc[:2]
        return roc, auc

    def save_results(self):
        """Function for saving the results."""
        # Save self.test_type_stats to a json file, make it json serializable
        # test_type_stats = {}
        # for k, v in self.test_type_stats.items():
        #     k: TestType
        #     v0 = v[0].__dict__
        #     v1 = v[1].__dict__
        #     k0 = k.value.capitalize()
        #     test_type_stats[k0] = (v0, v1)
        save_path = self.encoder_weights_path.parent.parent / "test_type_stats.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.test_type_stats, f)
