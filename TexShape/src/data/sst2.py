"""
Module to load the SST-2 dataset and preprocess it.
"""

from pathlib import Path
from typing import Tuple, Union

from datasets import load_dataset
import torch

from src.data.utils import (
    TexShapeDataset,
    extract_embeddings,
    load_mpnet_model_tokenizer,
)

SST2_TRAIN_TEST_SPLIT_RATIO = 0.9
SST2_SENT_LEN_THRESHOLD = 8


def load_sst2(
    *, data_loc: Union[Path, str], st_model_name: str, device=torch.device
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """
    Load the SST-2 dataset and split it into train and validation sets.

    :param data_loc: The path to the SST-2 dataset.

    :return: A tuple containing the train and validation datasets.
    """
    if isinstance(data_loc, str):
        data_loc = Path(data_loc)

    # Get the embeddings and labels from the model specific path
    data_loc = data_loc / "sst2" / st_model_name
    if not data_loc.exists() or not list(data_loc.iterdir()):
        make_dataset(
            device=device,
            data_loc=data_loc,
            st_model_name=st_model_name,
        )

    # Load the embeddings and labels
    train_embeddings = torch.load(data_loc / "embeddings.pt")
    train_sentiment_labels = torch.load(data_loc / "sentiment_labels.pt")
    train_sent_len_labels = torch.load(data_loc / "sent_len_labels.pt")

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


def make_dataset(data_loc: Path, st_model_name: str, device: torch.device) -> None:
    """
    :param device: torch.Tensor: Device to use for processing
    :param data_loc: Path: Path to save the processed data
    :param st_model_name: str: Sentence Transformer model name
    """
    model, tokenizer = load_mpnet_model_tokenizer(
        model_name=st_model_name, device=device
    )

    # Load the sst2 dataset
    dataset = load_dataset("stanfordnlp/sst2")
    dataset = dataset["train"]

    # Extract embeddings
    embeddings: torch.Tensor = extract_embeddings(
        dataset["sentence"], model, device=device
    )
    embeddings = embeddings.detach().cpu()

    # Get sentence length
    def get_sent_len(example):
        tokenized_sentence = tokenizer.tokenize(example["sentence"])
        sentence_length = len(tokenized_sentence)
        # Label sentence_length
        if sentence_length <= SST2_SENT_LEN_THRESHOLD:
            sent_len_label = 0
        else:
            sent_len_label = 1
        return {"sent_len": sent_len_label}

    # Map the function to the dataset
    dataset = dataset.map(get_sent_len)

    # Get sentiment labels and sentence length labels
    # pylint: disable=no-member
    sentiment_labels = torch.tensor(dataset["label"]).detach().cpu()
    sent_len_labels = torch.tensor(dataset["sent_len"]).detach().cpu()

    # Create the data directory
    data_loc.mkdir(parents=True, exist_ok=True)

    # Save the embeddings and labels
    torch.save(embeddings.detach().cpu(), data_loc / "embeddings.pt")
    torch.save(sentiment_labels.detach().cpu(), data_loc / "sentiment_labels.pt")
    torch.save(sent_len_labels.detach().cpu(), data_loc / "sent_len_labels.pt")
