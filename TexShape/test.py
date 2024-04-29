"""
Test the models on the test dataset.
Author: H. Kaan Kale
Email: hkaankale1@gmail.com
"""

from pathlib import Path
from typing import Tuple, Union, List, Dict
import logging

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils.general import (
    set_seed,
    configure_torch_backend,
)
from src.utils.config import (
    ExperimentParams,
    EncoderParams,
    ExperimentType,
    TestType,
    DatasetName,
    MNLIParams,
    MNLICombinationType,
    load_experiment_params,
)
from src.test.utility_privacy import UtilityPrivacyTester
from src.test.compression import CompressionTester
from src.data.utils import TexShapeDataset
from src.utils.testing import TestClass
from src.utils.general import get_roc_auc
from src.models.predict_model import SimpleClassifier
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model
from src.visualization.visualize import plot_accs
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



@hydra.main(config_path="configs", config_name="test_config", version_base="1.2")
def main(config: DictConfig) -> None:
    """Main function to test the models on the test dataset."""
    # Configure the logger
    logging.basicConfig(level=logging.DEBUG)

    # Number of epochs for training the test classifiers
    load_from_epoch: int = 5

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

    weights_dir = experiment_dir / "encoder_weights"
    if weights_dir is not None:
        if load_from_epoch is None:
            encoder_weights_path = sorted(
                weights_dir.glob("model_*.pt"), key=lambda x: int(x.stem.split("_")[-1])
            )[-1]
        else:
            encoder_weights_path = weights_dir / f"model_{load_from_epoch}.pt"

    # utility_privacy_tester = UtilityPrivacyTester(
    #     experiment_params=experiment_params,
    #     test_params=test_params,
    #     encoder_weights_path=encoder_weights_path,
    # )
    # utility_privacy_tester.run_all_tests()
    compression_tester = CompressionTester(
        experiment_params=experiment_params,
        test_params=test_params,
        encoder_weights_path=encoder_weights_path,
    )
    compression_tester.run_all_tests()

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


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
