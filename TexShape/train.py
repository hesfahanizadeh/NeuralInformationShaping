# Standard library imports
import math
import logging

# Third party imports
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

# Local imports
from src.models.models_to_train import Encoder
from src.models.utils import create_encoder_model
from src.utils.general import set_seed, configure_torch_backend, set_include_privacy, get_experiment_params, ExperimentParams
from src.data.utils import load_experiment_dataset, DatasetParams, get_dataset_params
from src.dual_optimization_encoder import DualOptimizationEncoder


@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)
    configure_torch_backend()
    logging.info(f"Seed: {seed}")

    device_idx: int = config.device_idx
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    experiment_params: ExperimentParams = get_experiment_params(config)
    dataset_params: DatasetParams = get_dataset_params(
        experiment_params.dataset_name, config
    )
    logging.info(experiment_params)
    logging.info(dataset_params)

    # Initialize encoder model
    encoder_model: Encoder = create_encoder_model(
        model_name=experiment_params.encoder_params.encoder_model_name,
        model_params=experiment_params.encoder_params.encoder_model_params,
    )
    encoder_model.to(device)
    logging.info(encoder_model)

    # Load the dataset
    dataset = load_experiment_dataset(
        dataset_params=dataset_params,
        device=device,
    )

    # Set if privacy goal is included
    include_privacy: bool = set_include_privacy(experiment_params.experiment_type)

    # Set the mine batch size
    if experiment_params.mine_params.mine_batch_size == -1:
        mine_batch_size = len(dataset)

    logging.info(f"Mine Batch Size: {mine_batch_size}")

    # Create a dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=mine_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize dual optimization model
    dual_optimization = DualOptimizationEncoder(
        experiment_params=experiment_params,
        encoder_model=encoder_model,
        data_loader=data_loader,
        device=device,
    )

    num_batches_final_MI = math.ceil(int(len(data_loader.dataset) / mine_batch_size))
    logging.info(f"Num batches Final MI: {num_batches_final_MI}")

    # Train the encoder
    dual_optimization.train_encoder(
        num_batches_final_MI=num_batches_final_MI,
        include_privacy=include_privacy,
        include_utility=True,
        gradient_batch_size=1,
    )


if __name__ == "__main__":
    main()
