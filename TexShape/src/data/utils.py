# Standard library imports
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# Third-party library imports
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs: torch.Tensor = inputs
        self.targets: torch.Tensor = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def load_sst2(
    train_data_dir_path: Path = Path("./data/sst2/train"),
    validation_data_dir_path: Path = Path("./data/sst2/val"),
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:

    train_embeddings_path = train_data_dir_path / "train_embeddings.pt"
    train_inputs: List[torch.Tensor] = torch.load(train_embeddings_path)

    train_public_labels_path = train_data_dir_path / "sentiment_labels.pt"
    train_public_labels = torch.load(train_public_labels_path)

    train_private_labels_path = train_data_dir_path / "sent_len_labels.pt"
    train_private_labels = torch.load(train_private_labels_path)

    val_embeddings_path = validation_data_dir_path / "val_embeddings.pt"
    val_inputs = torch.load(val_embeddings_path)

    val_public_labels_path = validation_data_dir_path / "sentiment_labels.pt"
    val_public_labels = torch.load(val_public_labels_path)

    val_private_labels_path = validation_data_dir_path / "sent_len_labels.pt"
    val_private_labels = torch.load(val_private_labels_path)

    return (train_inputs, train_public_labels, train_private_labels), (
        val_inputs,
        val_public_labels,
        val_private_labels,
    )


def load_mnli(
    device: torch.device,
    load_validation=False,
    embedding_model_name: str = "all-mpnet-base-v2",
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    # Load MultiNLI dataset
    dataset = load_dataset("multi_nli")

    # Load SentenceTransformer model
    model = SentenceTransformer(embedding_model_name)

    desired_genres = ["telephone", "government"]

    # Train Set
    train_filtered_dataset = dataset["train"].filter(
        lambda example: example["genre"] in desired_genres
    )

    train_targets_private = train_filtered_dataset["genre"]
    train_targets_private = torch.tensor(
        [0 if label == "government" else 1 for label in train_targets_private]
    )
    train_targets_public = torch.tensor(train_filtered_dataset["label"])

    # Encode
    train_original_premise = encode_dataset(
        train_filtered_dataset["premise"], model, device
    ).to(device)
    train_original_hypothesis = encode_dataset(
        train_filtered_dataset["hypothesis"], model, device
    ).to(device)

    if load_validation:
        # Validation Set
        validation_filtered_dataset = dataset["validation_matched"].filter(
            lambda example: example["genre"] in desired_genres
        )
        validation_original_premise = encode_dataset(
            validation_filtered_dataset["premise"], model, device
        ).to(device)
        validation_original_hypothesis = encode_dataset(
            validation_filtered_dataset["hypothesis"], model, device
        ).to(device)

        # Obtain private targets
        validation_targets_private = validation_filtered_dataset["genre"]
        validation_targets_private = torch.tensor(
            [0 if label == "government" else 1 for label in validation_targets_private]
        )

        validation_targets_public = torch.tensor(validation_filtered_dataset["label"])

    train_data = (
        train_original_premise,
        train_original_hypothesis,
        train_targets_public,
        train_targets_private,
    )

    if load_validation:
        validation_data = (
            validation_original_premise,
            validation_original_hypothesis,
            validation_targets_public,
            validation_targets_private,
        )
        return train_data, validation_data

    return train_data


def load_corona(train_data_path: Path, validation_data_path: Path) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Load the Corona dataset

    Args:
        train_data_path (Path): Path object to the training data
        validation_data_path (Path): Path object to the validation data

    Returns:
        Tuple[ Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ]:
        Tuple containing the training and validation data
    """
    # Load the training data
    # TODO: Fix type hinting
    train_dataset: Dict = torch.load(train_data_path)

    # Defining Input and Target Tensors for the training data
    train_inputs = torch.stack([v["encoded_text"] for v in train_dataset.values()])
    train_targets_public = torch.stack(
        [torch.tensor(v["country_label"]) for v in train_dataset.values()]
    )

    # Load the private labels for the training data
    train_targets_private = torch.stack(
        [torch.tensor(v["sentiment_label"]) for v in train_dataset.values()]
    )

    # Load the validation data
    val_dataset: Dict = torch.load(validation_data_path)

    # Defining Input and Target Tensors for the validation data
    validation_inputs = torch.stack([v["encoded_text"] for v in val_dataset.values()])
    validation_targets_public = torch.stack(
        [torch.tensor(v["country_label"]) for v in val_dataset.values()]
    )

    # Load the private labels for the validation data
    validation_targets_private = torch.stack(
        [torch.tensor(v["sentiment_label"]) for v in val_dataset.values()]
    )

    return (train_inputs, train_targets_public, train_targets_private), (
        validation_inputs,
        validation_targets_public,
        validation_targets_private,
    )


def encode_dataset(
    dataset, model: SentenceTransformer, device: torch.device, batch_size=128
) -> torch.Tensor:
    encoded = model.encode(
        dataset,
        convert_to_tensor=True,
        device=device,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).cpu()

    return encoded


def load_experiment_dataset(
    *,
    dataset_name: str,
    combination_type: str,
    experiment_type: str,
    device: Optional[torch.device] = None
) -> Tuple[CustomDataset, torch.Tensor, bool]:
    # TODO: Fix this model
    # Load dataset
    if dataset_name == "sst2":
        (train_inputs, train_public_labels, train_private_labels), _ = load_sst2()

    elif dataset_name == "mnli":
        if device is None:
            raise ValueError("Device cannot be None for MNLI dataset")
        (
            train_premise,
            train_hypothesis,
            train_public_labels,
            train_private_labels,
        ) = load_mnli(device, load_validation=False)

        if combination_type == "concat":
            train_inputs = torch.cat((train_premise, train_hypothesis), dim=-1)
        elif combination_type == "join":
            train_inputs = torch.cat((train_premise, train_hypothesis), dim=0)
            train_private_labels = torch.cat(
                (train_private_labels, train_private_labels), dim=0
            )
        elif combination_type == "premise_only":
            train_inputs = train_premise

    elif dataset_name == "corona":
        (train_inputs, train_public_labels, train_private_labels), _ = load_corona(
            "data/Corona_NLP/train_dict.pt",
            "data/Corona_NLP/test_dict.pt",
        )
    else:
        raise ValueError("Invalid dataset")

    # Training params
    if experiment_type == "utility":
        # utility_stats_network = FeedForwardMI3(encoded_embedding_size).to(device)
        # privacy_stats_network = FeedForwardMI3(encoded_embedding_size).to(device)
        dataset = CustomDataset(train_inputs, train_public_labels)
        include_privacy = False

    elif experiment_type == "utility+privacy":
        # utility_stats_network = FeedForwardMI3(encoded_embedding_size).to(device)
        # privacy_stats_network = FeedForwardMI3(encoded_embedding_size).to(device)
        dataset = CustomDataset(train_inputs, train_public_labels)
        include_privacy = True

    # Write if checks for all of this "utility", "utility+privacy", "compression", "compression+filtering"
    elif experiment_type == "compression":
        # utility_stats_network = FeedForwardMI(encoded_embedding_size, 768).to(device)
        # privacy_stats_network = FeedForwardMI3(encoded_embedding_size).to(device)
        dataset = CustomDataset(train_inputs, train_inputs)
        include_privacy = False

    elif experiment_type == "compression+filtering":
        # utility_stats_network = FeedForwardMI(encoded_embedding_size, 768).to(device)
        # privacy_stats_network = FeedForwardMI3(encoded_embedding_size).to(device)
        dataset = CustomDataset(train_inputs, train_inputs)
        include_privacy = True

    return dataset, train_private_labels, include_privacy
