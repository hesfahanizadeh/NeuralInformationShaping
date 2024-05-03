"""
Module to load the MNLI dataset and preprocess it.
"""

from pathlib import Path
from typing import Tuple, List

import torch
from datasets import load_dataset

from src.utils.config import MNLICombinationType
from src.data.utils import (
    TexShapeDataset,
    extract_embeddings,
    load_mpnet_model_tokenizer,
)


class MNLIDataset(TexShapeDataset):
    """MNLI Dataset"""

    def __init__(
        self,
        premise: torch.Tensor,
        hypothesis: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor,
    ):
        super().__init__(embeddings=premise, label1=label1, label2=label2)
        self.premise = self.embeddings
        self.hypothesis = hypothesis

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.premise[idx],
            self.hypothesis[idx],
            self.label1[idx],
            self.label2[idx],
        )


def load_mnli(
    *,
    data_loc: Path,
    st_model_name: str,
    device: torch.device,
    combination_type: MNLICombinationType,
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """
    Function to load the MNLI dataset and split it into train and validation sets.
    :param data_loc: The path to the location where datasets are stored.
    :param st_model_name: The name of the SentenceTransformer model.
    device (torch.device): The device to use for loading the model.
    :param combination_type: The type of combination to use for the embeddings.

    :return: A tuple containing the train and validation datasets.
    """
    data_loc = data_loc / "mnli" / st_model_name
    if not data_loc.exists() or not list(data_loc.iterdir()):
        make_dataset(
            data_loc=data_loc,
            st_model_name=st_model_name,
            device=device,
        )

    train_path = data_loc / "train"
    validation_path = data_loc / "validation"

    train_premise_embeddings = torch.load(train_path / "premise_embeddings.pt")
    train_hypothesis_embeddings = torch.load(train_path / "hypothesis_embeddings.pt")
    train_label1 = torch.load(train_path / "label.pt")
    train_label2 = torch.load(train_path / "genre_label.pt")

    validation_premise_embeddings = torch.load(
        validation_path / "premise_embeddings.pt"
    )
    validation_hypothesis_embeddings = torch.load(
        validation_path / "hypothesis_embeddings.pt",
    )
    validation_label1 = torch.load(validation_path / "label.pt")
    validation_label2 = torch.load(validation_path / "genre_label.pt")

    train_dataset = MNLIDataset(
        premise=train_premise_embeddings,
        hypothesis=train_hypothesis_embeddings,
        label1=train_label1,
        label2=train_label2,
    )

    validation_dataset = MNLIDataset(
        premise=validation_premise_embeddings,
        hypothesis=validation_hypothesis_embeddings,
        label1=validation_label1,
        label2=validation_label2,
    )
    train_dataset, validation_dataset = combine_embeddings(
        train_dataset,
        validation_dataset,
        combination_type=combination_type,
    )
    return train_dataset, validation_dataset


def combine_embeddings(
    train_dataset: MNLIDataset,
    validation_dataset: MNLIDataset,
    combination_type: MNLICombinationType,
):
    """
    Combine the premise and hypothesis embeddings based on the combination type.
    :param train_dataset: The training dataset.
    :param validation_dataset: The validation dataset.
    :param combination_type: The type of combination to use for the embeddings.
    :return: The processed train and validation datasets.
    """
    train_premise = train_dataset.premise
    train_hypothesis = train_dataset.hypothesis
    train_label1 = train_dataset.label1
    train_label2 = train_dataset.label2

    validation_premise = validation_dataset.premise
    validation_hypothesis = validation_dataset.hypothesis
    validation_label1 = validation_dataset.label1
    validation_label2 = validation_dataset.label2

    # Combine the premise and hypothesis embeddings
    # pylint: disable=no-member
    if combination_type == MNLICombinationType.CONCAT:
        train_embeddings = torch.cat((train_premise, train_hypothesis), dim=-1)
        validation_embeddings = torch.cat(
            (validation_premise, validation_hypothesis), dim=-1
        )

    elif combination_type == MNLICombinationType.JOIN:
        train_embeddings = torch.cat((train_premise, train_hypothesis), dim=0)
        train_label2 = torch.cat((train_label2, train_label2), dim=0)
        train_label1 = torch.cat((train_label1, train_label1), dim=0)

        validation_embeddings = torch.cat(
            (validation_premise, validation_hypothesis), dim=0
        )
        validation_label2 = torch.cat((validation_label2, validation_label2), dim=0)
        validation_label1 = torch.cat((validation_label1, validation_label1), dim=0)

    elif combination_type == MNLICombinationType.PREMISE_ONLY:
        train_embeddings = train_premise
        validation_embeddings = validation_premise

    train_dataset = TexShapeDataset(train_embeddings, train_label1, train_label2)
    validation_dataset = TexShapeDataset(
        validation_embeddings, validation_label1, validation_label2
    )
    return train_dataset, validation_dataset


def make_dataset(
    data_loc: Path,
    st_model_name: str,
    device: torch.device,
) -> None:
    """
    Function to create the embeddings, labels and save them to disk.
    :param data_loc: The path to the location where datasets are stored.
    :param st_model_name: The name of the SentenceTransformer model.
    :param device: The device to use for loading the model.
    """
    desired_genres: List[str] = ["telephone", "government"]

    # Load MultiNLI dataset
    dataset = load_dataset("multi_nli")

    # Load SentenceTransformer model
    model, _ = load_mpnet_model_tokenizer(model_name=st_model_name, device=device)

    # Train Set
    train_filtered_dataset = dataset["train"].filter(
        lambda example: example["genre"] in desired_genres
    )

    train_label1 = torch.tensor(train_filtered_dataset["label"])  # pylint: disable=no-member
    train_label2 = train_filtered_dataset["genre"]
    train_label2 = torch.tensor(  # pylint: disable=no-member
        [0 if label == "government" else 1 for label in train_label2]
    )

    train_premise_embeddings = extract_embeddings(
        train_filtered_dataset["premise"], model, device=device
    )

    train_hypothesis_embeddings = extract_embeddings(
        train_filtered_dataset["hypothesis"], model, device=device
    )

    # Validation Set
    validation_filtered_dataset = dataset["validation_matched"].filter(
        lambda example: example["genre"] in desired_genres
    )

    validation_label1 = torch.tensor(validation_filtered_dataset["label"])  # pylint: disable=no-member

    # Obtain private targets
    validation_label2 = validation_filtered_dataset["genre"]
    validation_label2 = torch.tensor(  # pylint: disable=no-member
        [0 if label == "government" else 1 for label in validation_label2]
    )

    validation_premise_embeddings = extract_embeddings(
        validation_filtered_dataset["premise"], model, device=device
    )

    validation_hypothesis_embeddings = extract_embeddings(
        validation_filtered_dataset["hypothesis"], model, device=device
    )

    # Create the data directories
    train_data_path = data_loc / "train"
    validation_data_path = data_loc / "validation"
    train_data_path.mkdir(parents=True, exist_ok=True)
    validation_data_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        train_premise_embeddings.detach().cpu(), train_data_path / "premise_embeddings.pt"
    )
    torch.save(
        train_hypothesis_embeddings.detach().cpu(), train_data_path / "hypothesis_embeddings.pt"
    )
    torch.save(train_label1.detach().cpu(), train_data_path / "label.pt")
    torch.save(train_label2.detach().cpu(), train_data_path / "genre_label.pt")

    torch.save(
        validation_premise_embeddings.detach().cpu(),
        validation_data_path / "premise_embeddings.pt",
    )
    torch.save(
        validation_hypothesis_embeddings.detach().cpu(),
        validation_data_path / "hypothesis_embeddings.pt",
    )
    torch.save(validation_label1.detach().cpu(), validation_data_path / "label.pt")
    torch.save(
        validation_label2.detach().cpu(), validation_data_path / "genre_label.pt"
    )
