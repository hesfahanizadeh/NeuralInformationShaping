# Standard library imports
import math
import logging
from pathlib import Path

# Third party imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig

# Local application imports
from src.models import models_to_train  
from src.models.models_to_train import Encoder
from src.utils.data_structures import ExperimentParams
from src.utils.experiment_setup import get_experiment_params
from src.data.utils import load_experiment_dataset
from src.dual_optimization_encoder import DualOptimizationEncoder

@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    logging.basicConfig(level=logging.INFO)
    
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)
    logging.info(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_params: ExperimentParams = get_experiment_params(config)
    logging.info(experiment_params)

    # Initialize encoder model
    encoder_model: Encoder = vars(models_to_train)[
        experiment_params.encoder_params.encoder_model_name
    ](**experiment_params.encoder_params.encoder_model_params)
    
    encoder_model.to(device)
    logging.info(encoder_model)
    
    embeddings_path: Path = Path(config.dataset.embeddings_path)

    dataset, train_private_labels, include_privacy = load_experiment_dataset(
        embeddings_path=embeddings_path,
        dataset_name=experiment_params.dataset_name,
        combination_type=experiment_params.combination_type,
        experiment_type=experiment_params.experiment_type,
        device=device,
    )

    # Set the mine batch size
    if experiment_params.mine_params.mine_batch_size == -1:
        mine_batch_size = len(train_private_labels)

    logging.info(f"Mine Batch Size: {mine_batch_size}")

    # Create a dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=mine_batch_size,
        shuffle=True,
        # num_workers=,
        # pin_memory=True,
    )

    # Initialize dual optimization model
    dual_optimization = DualOptimizationEncoder(
        experiment_params=experiment_params,
        encoder_model=encoder_model,
        data_loader=data_loader,
        device=device,
        private_labels=train_private_labels,
    )

    num_batches_final_MI = math.ceil(int(len(data_loader.dataset) / mine_batch_size))
    print("Num batches Final MI:", num_batches_final_MI)

    return
    # Train the encoder
    dual_optimization.train_encoder(
        num_batches_final_MI=num_batches_final_MI,
        include_privacy=include_privacy,
        include_utility=True,
        gradient_batch_size=1,
    )


if __name__ == "__main__":
    main()
