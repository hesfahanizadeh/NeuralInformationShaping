"""
General utility functions for the project.
Author: H. Kaan Kale
Email: hkaankale1@gmail.com
"""

import datetime
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_curve, auc

import numpy as np

from src.utils.config import DatasetName, DatasetParams, MNLIParams, ExperimentParams
from src.data.utils import TexShapeDataset
from src.data.sst2 import load_sst2
from src.data.corona import load_corona
from src.data.mnli import load_mnli

SENTENCE_EMBEDDING_DIM = {
    "bert-base-uncased": 768,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
}


def load_dataset(
    *,
    dataset_params: DatasetParams,
    device: torch.device,
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """
    Load the train and validation datasets.
    :param dataset_params: The parameters for the dataset.
    :param device: The device to use for loading the model.
    """
    data_loc = dataset_params.dataset_loc
    dataset_name = dataset_params.dataset_name

    if dataset_name == DatasetName.SST2:
        train_dataset, validation_dataset = load_sst2(
            data_loc=data_loc,
            st_model_name=dataset_params.st_model_name,
            device=device,
        )

    elif dataset_name == DatasetName.MNLI:
        mnli_params: MNLIParams = dataset_params
        train_dataset, validation_dataset = load_mnli(
            data_loc=data_loc,
            st_model_name=dataset_params.st_model_name,
            device=device,
            combination_type=mnli_params.combination_type,
        )

    elif dataset_name == DatasetName.CORONA:
        train_dataset, validation_dataset = load_corona(
            data_loc=data_loc, st_model_name=dataset_params.st_model_name, device=device
        )
    else:
        raise ValueError("Invalid dataset")

    return train_dataset, validation_dataset


def check_experiment_params(experiment_params: ExperimentParams):
    # TODO: Finish this function
    encoder_model_params = experiment_params.encoder_params.encoder_model_params
    st_model_name = experiment_params.dataset_params.st_model_name
    st_model_dim = SENTENCE_EMBEDDING_DIM[st_model_name]

    assert encoder_model_params["in_dim"] == st_model_dim, "Input dimension mismatch"

    # Get the experiment type
    raise NotImplementedError


# ROC and AUC functions # TODO Optimize this function
def get_roc_auc(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model = model.to(device)
    model = model.eval()

    # Get the predictions and labels
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(outputs.cpu().numpy()[:, 1])

    roc = roc_curve(all_labels, all_confidences)

    # Get the AUC
    auc_score = auc(roc[0], roc[1])
    return roc, auc_score


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed)


def configure_torch_backend():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_date() -> str:
    return datetime.datetime.now().strftime("%m_%d_%y")
