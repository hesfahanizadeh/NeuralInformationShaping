"""
Module to load the Corona dataset and preprocess it.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


from tqdm import tqdm
import torch

from src.data.utils import load_mpnet_model_tokenizer, TexShapeDataset

RAW_DATA_PATH = Path("TexShape/data/raw")


def load_corona(
    data_loc: Path,
    device: torch.device,
    st_model_name: str,
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """Load the Corona dataset"""
    data_loc = data_loc / "corona" / st_model_name
    if not data_loc.exists() or not list(data_loc.iterdir()):
        make_dataset(
            device=device,
            data_loc=data_loc,
            st_model_name=st_model_name,
        )

    train_data_path = data_loc / "train"
    validation_data_path = data_loc / "validation"

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


def make_dataset(
    device: torch.device,
    data_loc: Path = Path("data/processed/corona"),
    st_model_name: str = "all-mpnet-base-v2",
) -> None:
    """
    Create the processed data for the Corona dataset
    :param: device: int: Device to use for processing
    :return: None
    """
    train_raw_data = RAW_DATA_PATH / "corona" / "train.csv"
    validation_raw_data = RAW_DATA_PATH / "corona" / "validation.csv"

    # Read the raw data as a dictionary
    train_df = pd.read_csv(train_raw_data)
    validation_df = pd.read_csv(validation_raw_data)

    # Load SentenceTransformer model
    model, _ = load_mpnet_model_tokenizer(model_name=st_model_name, device=device)

    train_dict = {}
    validation_dict = {}

    for i, sample in tqdm(train_df.iterrows()):
        text = sample["text_clean"]
        sentiment_label = sample["Sentiment"]
        country_label = sample["Country"]
        text_embedding = (
            model.encode(
                text,
                convert_to_tensor=True,
                device=device,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            .detach()
            .cpu()
        )

        train_dict[i] = {
            "embedding": text_embedding,
            "sentiment_label": sentiment_label,
            "country_label": country_label,
        }

    for i, sample in tqdm(validation_df.iterrows()):
        text = sample["text_clean"]
        sentiment_label = sample["Sentiment"]
        country_label = sample["Country"]
        text_embedding = (
            model.encode(
                text,
                convert_to_tensor=True,
                device=device,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            .detach()
            .cpu()
        )

        validation_dict[i] = {
            "embedding": text_embedding,
            "sentiment_label": sentiment_label,
            "country_label": country_label,
        }

    # Defining Input and Target Tensors for the training data
    # pylint: disable=no-member
    train_embeddings = torch.stack([v["embedding"] for v in train_dict.values()])
    train_country_label = torch.stack(
        [torch.tensor(v["country_label"]) for v in train_dict.values()]
    )

    # Load the private labels for the training data
    train_sentiment_label = torch.stack(
        [torch.tensor(v["sentiment_label"]) for v in train_dict.values()]
    )

    # Defining Input and Target Tensors for the validation data
    validation_embeddings = torch.stack(
        [v["embedding"] for v in validation_dict.values()]
    )
    validation_country_label = torch.stack(
        [torch.tensor(v["country_label"]) for v in validation_dict.values()]
    )

    # Load the private labels for the validation data
    validation_sentiment_label = torch.stack(
        [torch.tensor(v["sentiment_label"]) for v in validation_dict.values()]
    )

    # Create the necessary directories
    train_data_path = data_loc / "train"
    validation_data_path = data_loc / "validation"
    train_data_path.mkdir(parents=True, exist_ok=True)
    validation_data_path.mkdir(parents=True, exist_ok=True)

    # Save the training data
    torch.save(train_embeddings.detach().cpu(), train_data_path / "embeddings.pt")
    torch.save(
        train_country_label.detach().cpu(), train_data_path / "country_labels.pt"
    )
    torch.save(
        train_sentiment_label.detach().cpu(), train_data_path / "sentiment_labels.pt"
    )

    # Save the validation data
    torch.save(
        validation_embeddings.detach().cpu(), validation_data_path / "embeddings.pt"
    )
    torch.save(
        validation_country_label.detach().cpu(),
        validation_data_path / "country_labels.pt",
    )
    torch.save(
        validation_sentiment_label.detach().cpu(),
        validation_data_path / "sentiment_labels.pt",
    )


if __name__ == "__main__":
    print("Corona Dataset")
