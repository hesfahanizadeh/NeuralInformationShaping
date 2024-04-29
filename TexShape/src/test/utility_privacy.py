"""Utility privacy test module."""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import logging

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import numpy as np

from src.utils.config import ExperimentParams, EncoderParams, TestType
from src.data.utils import load_train_dataset, TexShapeDataset
from src.utils.testing import TestClass
from src.utils.general import get_roc_auc
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model
from src.test.utils import TestParams


@dataclass
class TestStats:
    """Test stats dataclass."""

    validation_roc: np.ndarray
    validation_auc_score: float
    validation_acc: float
    train_acc: float


TEST_TYPES = [
    # TestType.RANDOM,
    # TestType.ORIGINAL,
    TestType.TEXSHAPE,
    # TestType.NOISE,
]


class UtilityPrivacyTester:
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
        self.num_class = 2
        self.noise_std = 0.1
        self.test_type_stats = {}
        self.encoder_weights_path = encoder_weights_path

    def run_all_tests(self):
        """Run all tests."""
        for test_type in TEST_TYPES:
            self.run_test_type(test_type)

    def run_test_type(self, test_type: TestType):
        """Run the test for the given test type."""
        max_epoch = self.test_params.max_epoch
        logging.info("Testing model for test type: %s", test_type)
        (
            train_utility_dataloader,
            validation_utility_dataloader,
            train_private_dataloader,
            validation_private_dataloader,
        ) = self.create_dataloaders(test_type=test_type)

        logging.info("Training the utility model")
        utility_stats = self.train_and_evaluate_model(
            test_type,
            train_utility_dataloader,
            validation_utility_dataloader,
            max_epoch,
        )

        logging.info("Training the privacy model")
        privacy_results = self.train_and_evaluate_model(
            test_type,
            train_private_dataloader,
            validation_private_dataloader,
            max_epoch,
        )

        self.test_type_stats[test_type] = (utility_stats, privacy_results)

    def train_and_evaluate_model(
        self,
        test_type: TestType,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        max_epochs: int,
    ) -> TestStats:
        """Train and evaluate the model."""
        classifier_model: SimpleClassifier = self.load_test_model(test_type=test_type)
        pl_model = TestClass(model=classifier_model, num_class=self.num_class)
        trainer = self.train_model(
            pl_model=pl_model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            max_epochs=max_epochs,
        )
        roc_auc_results = self.get_roc_auc(classifier_model, validation_dataloader)
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
        logging.info("Training model")
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

    def create_utility_dataset(
        self, train_dataset: TexShapeDataset, validation_dataset: TexShapeDataset
    ):
        train_utility_dataset = TensorDataset(
            train_dataset.embeddings, train_dataset.label1
        )
        validation_utility_dataset = TensorDataset(
            validation_dataset.embeddings, validation_dataset.label1
        )
        return train_utility_dataset, validation_utility_dataset

    def create_private_dataset(
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

        utility_datasets = self.create_utility_dataset(
            train_dataset, validation_dataset
        )
        train_utility_dataloader = DataLoader(
            dataset=utility_datasets[0], batch_size=batch_size, shuffle=True
        )
        validation_utility_dataloader = DataLoader(
            dataset=utility_datasets[1],
            batch_size=batch_size,
            shuffle=False,
        )
        private_datasets = self.create_private_dataset(
            train_dataset, validation_dataset
        )
        train_private_dataloader = DataLoader(
            dataset=private_datasets[0], batch_size=batch_size, shuffle=True
        )
        validation_private_dataloader = DataLoader(
            dataset=private_datasets[1],
            batch_size=batch_size,
            shuffle=False,
        )

        return (
            train_utility_dataloader,
            validation_utility_dataloader,
            train_private_dataloader,
            validation_private_dataloader,
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
        if test_type == TestType.NOISE:
            train_dataset.add_noise_embedding(self.noise_std)
            return train_dataset, validation_dataset

        raise ValueError(f"Test type {test_type} not supported.")

    def load_test_model(self, test_type: str) -> SimpleClassifier:
        """Load the test model."""
        original_in_dim = 768
        out_dim = 2
        encoder_params: EncoderParams = self.experiment_params.encoder_params
        texshape_in_dim: int = encoder_params.encoder_model_params["out_dim"]

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
        else:
            raise ValueError(
                f"Test type {test_type} not supported. Supported types are {TestType}"
            )

        model.eval()
        return model

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
        train_embeddings = encoder_model(train_embeddings)
        validation_embeddings = encoder_model(validation_embeddings)

        return train_embeddings.detach().cpu(), validation_embeddings.detach().cpu()

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
