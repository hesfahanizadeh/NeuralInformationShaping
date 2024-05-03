"""
Train the encoder model using dual optimization.
Author: H. Kaan Kale
Email: hkaankale1@gmail.com
"""

# Standard library imports
import math
import logging
from pathlib import Path

# Third party imports
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Local imports
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model
from src.utils.general import set_seed, configure_torch_backend, load_dataset
from src.utils.config import (
    set_include_privacy,
    load_experiment_params,
    ExperimentParams,
)
from src.data.utils import configure_dataset_for_experiment_type
from src.dual_optimization_encoder import DualOptimizationEncoder


@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """Main function to train the encoder model using dual optimization."""
    logging.basicConfig(level=logging.INFO)

    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)
    configure_torch_backend()
    logging.info("Seed: %s", seed)

    device_idx: int = config.device_idx
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    experiment_params: ExperimentParams = load_experiment_params(config)

    # Get the experiment directory path
    experiment_dir_path: Path = Path(HydraConfig.get().runtime.output_dir)
    logging.info("Experiment Directory Path: %s", experiment_dir_path)
    logging.info(experiment_params)

    # Initialize encoder model
    encoder_model: Encoder = create_encoder_model(
        model_name=experiment_params.encoder_params.encoder_model_name,
        model_params=experiment_params.encoder_params.encoder_model_params,
    )
    encoder_model.to(device)
    logging.info(encoder_model)

    # Load the dataset
    dataset, _ = load_dataset(
        dataset_params=experiment_params.dataset_params,
        device=device,
    )

    # Configure the dataset according to the expeirment type
    dataset = configure_dataset_for_experiment_type(
        dataset, experiment_params.experiment_type
    )

    # Set if privacy goal is included TODO: Delete this handle from the params of the encoder
    include_privacy: bool = set_include_privacy(experiment_params.experiment_type)

    # Set the mine batch size
    if experiment_params.mine_params.mine_batch_size == -1:
        mine_batch_size = len(dataset)

    logging.info("Mine Batch Size: %s", mine_batch_size)

    # Create a dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=mine_batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    # Initialize dual optimization model
    dual_optimization = DualOptimizationEncoder(
        experiment_params=experiment_params,
        encoder_model=encoder_model,
        data_loader=data_loader,
        device=device,
        experiment_dir_path=experiment_dir_path,
        device_idx=device_idx,
    )

    num_batches_final_mi = math.ceil(int(len(dataset) / mine_batch_size))
    logging.info("Num batches Final MI: %s", num_batches_final_mi)

    # Train the encoder
    dual_optimization.train_encoder(
        num_batches_final_MI=num_batches_final_mi,
        include_privacy=include_privacy,
        include_utility=True,
        gradient_batch_size=1,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
