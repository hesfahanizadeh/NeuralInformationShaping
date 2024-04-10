# Standard library imports
from pathlib import Path
from typing import Tuple

# Third-party library imports
import torch


class TexShapeDataset(torch.utils.data.Dataset):
    def __init__(
        self, embeddings: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor
    ):
        self.embeddings = embeddings
        self.label1 = label1
        self.label2 = label2

    def switch_labels(self) -> None:
        self.label1, self.label2 = self.label2, self.label1
        
    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.label1[idx], self.label2[idx]


class MNLI_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        premise: torch.Tensor,
        hypothesis: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor,
    ):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label1 = label1
        self.label2 = label2

    def __len__(self) -> int:
        return len(self.premise)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.premise[idx],
            self.hypothesis[idx],
            self.label1[idx],
            self.label2[idx],
        )

def load_sst2(
    sst2_data_path: Path = Path("data/processed/sst2"),
) -> Tuple[TexShapeDataset, TexShapeDataset]:

    train_dataset: TexShapeDataset = torch.load(sst2_data_path / "train.pt")
    validation_dataset: TexShapeDataset = torch.load(sst2_data_path / "validation.pt")
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


def load_experiment_dataset(
    *,
    dataset_name: str,
    data_path: Path = Path("data/processed"),
    combination_type: str = "concat",
) -> TexShapeDataset:
    if dataset_name == "sst2":
        data_path = data_path / "sst2"
        train_dataset, _ = load_sst2(data_path=data_path)

    elif dataset_name == "mnli":
        train_dataset: MNLI_Dataset
        data_path = data_path / "mnli"
        train_dataset, _ = load_mnli(data_path=data_path)
        train_premise = train_dataset.premise
        train_hypothesis = train_dataset.hypothesis
        train_label1 = train_dataset.label1
        train_label2 = train_dataset.label2

        if combination_type == "concat":
            train_embeddings = torch.cat((train_premise, train_hypothesis), dim=-1)

        elif combination_type == "join":
            train_embeddings = torch.cat((train_premise, train_hypothesis), dim=0)
            train_label2 = torch.cat((train_label2, train_label2), dim=0)

        elif combination_type == "premise_only":
            train_embeddings = train_premise

        train_dataset = TexShapeDataset(train_embeddings, train_label1, train_label2)

    elif dataset_name == "corona":
        data_path = data_path / "corona"
        train_dataset, _ = load_corona(data_path=data_path)
    else:
        raise ValueError("Invalid dataset")

    return train_dataset
