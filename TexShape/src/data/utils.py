# Standard library imports
from pathlib import Path
from typing import Tuple

# Third-party library imports
import torch

from omegaconf import DictConfig

from src.utils.general import SST2_Params, MNLI_Params, DatasetParams
from src.data.utils import preprocess_sst2, preprocess_mnli, preprocess_corona

class TexShapeDataset(torch.utils.data.Dataset):
    def __init__(
        self, embeddings: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor
    ):
        self.embeddings = embeddings
        self.label1 = label1
        self.label2 = label2
        self.embedding_shape = embeddings.shape
        
    def switch_labels(self) -> None:
        self.label1, self.label2 = self.label2, self.label1
        
    def create_noise_embedding(self, stddev) -> torch.Tensor:
        return torch.normal(0, stddev, size=self.embedding_shape)
    
    def add_noise_embedding(self, stddev)-> None:
        noise = self.create_noise_embedding(stddev)
        self.embeddings = self.embeddings + noise
        
    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.label1[idx], self.label2[idx]


class MNLI_Dataset(TexShapeDataset):
    def __init__(
        self,
        premise: torch.Tensor,
        hypothesis: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor,
    ):
        super().__init__(embeddings=premise, label1=label1, label2=label2)
        self.hypothesis = hypothesis

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.embeddings[idx],
            self.hypothesis[idx],
            self.label1[idx],
            self.label2[idx],
        )


def load_test_dataset(dataset_name: str, dataset_params: DatasetParams):
    if dataset_name == "sst2":
        dataset_params: SST2_Params = dataset_params
        train_dataset, validation_dataset = load_sst2(
            sst2_data_path=dataset_params.dataset_loc, train_test_split_ratio=dataset_params.train_test_split_ratio
        )
        
    else:
        raise ValueError("Invalid dataset name")
    
    return train_dataset, validation_dataset


def load_experiment_dataset(
    *,
    dataset_params: DatasetParams,
    device: torch.device,
) -> TexShapeDataset:
    data_loc = dataset_params.dataset_loc
    dataset_name = dataset_params.dataset_name

    if not data_loc.exists():
        data_loc.mkdir(parents=True, exist_ok=True)

    if dataset_name == "sst2":
        sst2_dataset_params: SST2_Params = dataset_params
        # Check if path exists
        dataset_path: Path = data_loc / "dataset.pt"

        if not dataset_path.exists():
            preprocess_sst2(device=device, data_path=data_loc)

        train_test_split_ratio: str = sst2_dataset_params.train_test_split_ratio
        train_dataset, _ = load_sst2(
            sst2_data_path=data_loc, train_test_split_ratio=train_test_split_ratio
        )

    elif dataset_name == "mnli":
        train_dataset: MNLI_Dataset
        mnli_dataset_params: MNLI_Params = dataset_params

        data_loc = data_loc / "mnli"
        train_dataset, _ = load_mnli(data_path=data_loc)
        train_premise = train_dataset.premise
        train_hypothesis = train_dataset.hypothesis
        train_label1 = train_dataset.label1
        train_label2 = train_dataset.label2

        combination_type = mnli_dataset_params.combination_type
        if combination_type == "concat":
            train_embeddings = torch.cat((train_premise, train_hypothesis), dim=-1)

        elif combination_type == "join":
            train_embeddings = torch.cat((train_premise, train_hypothesis), dim=0)
            train_label2 = torch.cat((train_label2, train_label2), dim=0)

        elif combination_type == "premise_only":
            train_embeddings = train_premise

        train_dataset = TexShapeDataset(train_embeddings, train_label1, train_label2)

    elif dataset_name == "corona":
        data_loc = data_loc / "corona"
        train_dataset, _ = load_corona(data_path=data_loc)
    else:
        raise ValueError("Invalid dataset")

    return train_dataset




def load_dataset_params(dataset_name: str, config: DictConfig) -> DatasetParams:
    dataset_params = config.dataset
    if dataset_name == "sst2":
        return SST2_Params(**dataset_params)
    raise ValueError("Invalid dataset name")



def load_sst2(
    sst2_data_path: Path, train_test_split_ratio: float = 0.9
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """
    Load the SST-2 dataset and split it into train and validation sets.

    :param sst2_data_path: The path to the SST-2 dataset.
    :param train_test_split_ratio: The ratio to split the dataset into train and validation sets. Default is 0.9.

    :return: A tuple containing the train and validation datasets.
    """
    dataset: TexShapeDataset = torch.load(sst2_data_path / "dataset.pt")

    # Train Test Split
    train_size = int(train_test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataset = train_dataset.dataset
    validation_dataset = validation_dataset.dataset
    return train_dataset, validation_dataset


def load_mnli(
    mnli_data_path: Path = Path("data/processed/mnli"),
) -> Tuple[MNLI_Dataset, MNLI_Dataset]:
    train_dataset: MNLI_Dataset = torch.load(mnli_data_path / "train.pt")
    validation_dataset: MNLI_Dataset = torch.load(mnli_data_path / "validation.pt")
    return train_dataset, validation_dataset


def load_corona(
    data_path: Path = Path("data/processed/corona"),
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    train_dataset: TexShapeDataset = torch.load(data_path / "train.pt")
    validation_dataset: TexShapeDataset = torch.load(data_path / "validation.pt")
    return train_dataset, validation_dataset


