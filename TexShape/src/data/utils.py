# Standard library imports
from pathlib import Path
from typing import Tuple, Union

# Third-party library imports
import torch

from src.utils.config import (
    # SST2Params,
    MNLIParams,
    DatasetParams,
    DatasetName,
    MNLICombinationType,
    ExperimentType,
)
from src.data.preprocess import preprocess_sst2, preprocess_mnli, preprocess_corona

SST2_TRAIN_TEST_SPLIT_RATIO = 0.9


class TexShapeDataset(torch.utils.data.Dataset):
    def __init__(
        self, embeddings: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor
    ):
        self.embeddings = embeddings
        self.label1 = label1
        self.label2 = label2
        self.num_class1 = len(torch.unique(label1))
        self.num_class2 = len(torch.unique(label2))
        self.embedding_shape = embeddings.shape

    def switch_labels(self) -> None:
        self.label1, self.label2 = self.label2, self.label1

    def create_noise_embedding(self, stddev) -> torch.Tensor:
        return torch.normal(0, stddev, size=self.embedding_shape)  # pylint: disable=no-member

    def add_noise_embedding(self, stddev) -> None:
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


def load_train_dataset(
    *,
    dataset_params: DatasetParams,
    device: torch.device,
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    data_loc = dataset_params.dataset_loc
    dataset_name = dataset_params.dataset_name

    if not data_loc.exists():
        data_loc.mkdir(parents=True, exist_ok=True)

    if dataset_name == DatasetName.SST2:
        # Check if path exists
        train_embedding_path: Path = data_loc / "embeddings.pt"

        if not train_embedding_path.exists():
            preprocess_sst2(device=device, data_path=data_loc)

        train_dataset, validation_dataset = load_sst2(sst2_data_path=data_loc)

    elif dataset_name == DatasetName.MNLI:
        train_dataset, validation_dataset = process_mnli_for_train(
            data_loc=data_loc, dataset_params=dataset_params, device=device
        )

    elif dataset_name == DatasetName.CORONA:
        train_embedding_path: Path = data_loc / "train" / "embeddings.pt"
        if not train_embedding_path.exists():
            preprocess_corona(device=device)

        train_dataset, validation_dataset = load_corona(data_path=data_loc)
    else:
        raise ValueError("Invalid dataset")

    # train_dataset.embeddings = train_dataset.embeddings.cpu()
    # train_dataset.label1 = train_dataset.label1.cpu()
    # train_dataset.label2 = train_dataset.label2.cpu()

    # validation_dataset.embeddings = validation_dataset.embeddings.cpu()
    # validation_dataset.label1 = validation_dataset.label1.cpu()
    # validation_dataset.label2 = validation_dataset.label2.cpu()
    return train_dataset, validation_dataset


def load_test_dataset(
    *,
    dataset_params: DatasetParams,
    device: torch.device,
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    data_loc = dataset_params.dataset_loc
    dataset_name = dataset_params.dataset_name

    if not data_loc.exists():
        data_loc.mkdir(parents=True, exist_ok=True)

    if dataset_name == DatasetName.SST2:
        # Check if path exists
        train_embedding_path: Path = data_loc / "embeddings.pt"

        if not train_embedding_path.exists():
            preprocess_sst2(device=device, data_path=data_loc)

        train_dataset, validation_dataset = load_sst2(sst2_data_path=data_loc)

    elif dataset_name == DatasetName.MNLI:
        train_dataset, validation_dataset = load_mnli(mnli_data_path=data_loc)

    elif dataset_name == DatasetName.CORONA:
        train_embedding_path: Path = data_loc / "train" / "embeddings.pt"
        if not train_embedding_path.exists():
            preprocess_corona(device=device)

        train_dataset, validation_dataset = load_corona(data_path=data_loc)
    else:
        raise ValueError("Invalid dataset")

    return train_dataset, validation_dataset


def process_mnli_for_train(
    data_loc: Path, dataset_params: MNLIParams, device: torch.device
):
    train_embedding_path: Path = data_loc / "train" / "premise_embeddings.pt"
    if not train_embedding_path.exists():
        preprocess_mnli(device=device)

    train_dataset: MNLI_Dataset
    mnli_dataset_params: MNLIParams = dataset_params

    train_dataset: MNLI_Dataset
    validation_dataset: MNLI_Dataset
    train_dataset, validation_dataset = load_mnli(mnli_data_path=data_loc)

    train_premise = train_dataset.premise
    train_hypothesis = train_dataset.hypothesis
    train_label1 = train_dataset.label1
    train_label2 = train_dataset.label2

    validation_premise = validation_dataset.premise
    validation_hypothesis = validation_dataset.hypothesis
    validation_label1 = validation_dataset.label1
    validation_label2 = validation_dataset.label2

    # Get the combination type
    combination_type = mnli_dataset_params.combination_type

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


def load_sst2(
    sst2_data_path: Union[Path, str],
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """
    Load the SST-2 dataset and split it into train and validation sets.

    :param sst2_data_path: The path to the SST-2 dataset.

    :return: A tuple containing the train and validation datasets.
    """
    if isinstance(sst2_data_path, str):
        sst2_data_path = Path(sst2_data_path)

    train_embeddings = torch.load(sst2_data_path / "embeddings.pt")
    train_sentiment_labels = torch.load(sst2_data_path / "sentiment_labels.pt")
    train_sent_len_labels = torch.load(sst2_data_path / "sent_len_labels.pt")

    dataset = TexShapeDataset(
        embeddings=train_embeddings,
        label1=train_sentiment_labels,
        label2=train_sent_len_labels,
    )
    train_size = int(SST2_TRAIN_TEST_SPLIT_RATIO * len(dataset))
    test_size = len(dataset) - train_size

    train_indices, validation_indices = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Function to extract data based on indices from the original dataset
    def create_subset_dataset(original_dataset, subset_indices):
        subset_embeddings = [
            original_dataset.embeddings[i] for i in subset_indices.indices
        ]
        subset_label1 = [original_dataset.label1[i] for i in subset_indices.indices]
        subset_label2 = [original_dataset.label2[i] for i in subset_indices.indices]

        # pylint: disable=no-member
        return TexShapeDataset(
            embeddings=torch.stack(subset_embeddings),
            label1=torch.tensor(subset_label1),
            label2=torch.tensor(subset_label2),
        )

    # Create the new TexShape datasets for train and validation
    train_dataset = create_subset_dataset(dataset, train_indices)
    validation_dataset = create_subset_dataset(dataset, validation_indices)

    return train_dataset, validation_dataset


def load_mnli(
    mnli_data_path: Path = Path("data/processed/mnli"),
) -> Tuple[MNLI_Dataset, MNLI_Dataset]:
    train_path = mnli_data_path / "train"
    validation_path = mnli_data_path / "validation"

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

    train_dataset = MNLI_Dataset(
        premise=train_premise_embeddings,
        hypothesis=train_hypothesis_embeddings,
        label1=train_label1,
        label2=train_label2,
    )

    validation_dataset = MNLI_Dataset(
        premise=validation_premise_embeddings,
        hypothesis=validation_hypothesis_embeddings,
        label1=validation_label1,
        label2=validation_label2,
    )
    return train_dataset, validation_dataset


def load_corona(
    data_path: Path = Path("data/processed/corona/"),
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """Load the Corona dataset"""
    train_data_path = data_path / "train"
    validation_data_path = data_path / "validation"

    # Load the training data
    train_embeddings = torch.load(train_data_path / "embeddings.pt")
    train_country_label = torch.load(train_data_path / "country_labels.pt")
    train_sentiment_label = torch.load(train_data_path / "sentiment_labels.pt")

    # Load the validation data
    validation_embeddings = torch.load(validation_data_path / "embeddings.pt")
    validation_country_label = torch.load(validation_data_path / "country_labels.pt")
    validation_sentiment_label = torch.load(
        validation_data_path / "sentiment_labels.pt"
    )

    train_dataset = TexShapeDataset(
        embeddings=train_embeddings,
        label1=train_country_label,
        label2=train_sentiment_label,
    )

    validation_dataset = TexShapeDataset(
        embeddings=validation_embeddings,
        label1=validation_country_label,
        label2=validation_sentiment_label,
    )

    return train_dataset, validation_dataset


def configure_dataset_for_experiment_type(
    dataset: TexShapeDataset, experiment_type: ExperimentType
) -> TexShapeDataset:
    """
    Make the necessary configurations to the dataset based on the experiment type.
    :param dataset: The dataset to configure.
    :param experiment_type: The type of experiment to configure the dataset for.
    """
    if experiment_type == ExperimentType.UTILITY:
        pass
    elif experiment_type == ExperimentType.UTILITY_PRIVACY:
        pass
    elif experiment_type == ExperimentType.COMPRESSION:
        dataset.label1 = dataset.embeddings.clone()
    elif experiment_type == ExperimentType.COMPRESSION_PRIVACY:
        dataset.label1 = dataset.embeddings.clone()
    else:
        raise ValueError("Invalid experiment type")

    return dataset
