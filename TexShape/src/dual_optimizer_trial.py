import math
import numpy as np

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything

# from utils.data import load_sst2, load_mnli, load_corona
from models import models_to_train
from utils.data_structures import ExperimentParams
from utils.experiment_setup import get_experiment_params
from data.utils import load_experiment_dataset
from dual_optimization_encoder import DualOptimizationEncoder

def main():
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_params: ExperimentParams = get_experiment_params()
    print(experiment_params)

    dataset, train_private_labels, include_privacy = load_experiment_dataset(
        dataset_name=experiment_params.dataset_name,
        combination_type=experiment_params.combination_type,
        experiment_type=experiment_params.experiment_type,
        device=device,
    )

    # Set the mine batch size
    if experiment_params.mine_args.mine_batch_size == -1:
        mine_batch_size = len(train_private_labels)

    print("Mine Batch Size:", mine_batch_size)

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
        encoder_model=models_to_train.Encoder(),
        data_loader=data_loader,
        device=device,
        private_labels=train_private_labels,
    )

    num_batches_final_MI = math.ceil(int(len(data_loader.dataset) / mine_batch_size))
    print("Num batches Final MI:", num_batches_final_MI)
    
    # Train the encoder
    dual_optimization.train_encoder(
        num_batches_final_MI=num_batches_final_MI,
        include_privacy=include_privacy,
        include_utility=True,
        gradient_batch_size=1,
    )


if __name__ == "__main__":
    main()