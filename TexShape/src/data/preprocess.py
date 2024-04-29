from pathlib import Path
from typing import List

import torch
import transformers
from transformers import BertTokenizer, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

import pandas as pd
from tqdm import tqdm

RAW_DATA_PATH = Path("data/raw")
SST2_SENT_LEN_THRESHOLD = 8


def preprocess_sst2(device: torch.Tensor, data_path: Path) -> None:
    model_name: str = "all-mpnet-base-v2"

    tokenizer = load_mpnet_tokenizer()
    model = load_mpnet_model(model_name=model_name, device=device)

    dataset = load_dataset("stanfordnlp/sst2")
    dataset = dataset["train"]

    embeddings: torch.Tensor = extract_embeddings(
        dataset["sentence"], model, device=device
    )
    embeddings = embeddings.detach().cpu()

    def get_sent_len(example):
        tokenized_sentence = tokenizer.tokenize(example["sentence"])
        sentence_length = len(tokenized_sentence)
        # Label sentence_length
        if sentence_length <= SST2_SENT_LEN_THRESHOLD:
            sent_len_label = 0
        else:
            sent_len_label = 1
        return {"sent_len": sent_len_label}

    dataset = dataset.map(get_sent_len)

    # pylint: disable=no-member
    sentiment_labels = torch.tensor(dataset["label"]).detach().cpu()
    sent_len_labels = torch.tensor(dataset["sent_len"]).detach().cpu()

    # Check if path exists
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    torch.save(embeddings.cpu(), data_path / "embeddings.pt")
    torch.save(sentiment_labels, data_path / "sentiment_labels.pt")
    torch.save(sent_len_labels, data_path / "sent_len_labels.pt")


def preprocess_mnli(
    device: torch.device,
    data_path: Path = Path("data/processed/mnli"),
) -> None:
    model_name: str = "all-mpnet-base-v2"
    desired_genres: List[str] = ["telephone", "government"]

    # Load MultiNLI dataset
    dataset = load_dataset("multi_nli")

    # Load SentenceTransformer model
    model = load_mpnet_model(model_name=model_name, device=device)

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

    train_data_path = data_path / "train"
    validation_data_path = data_path / "validation"

    # Check if path exists
    if not train_data_path.exists():
        train_data_path.mkdir(parents=True, exist_ok=True)

    if not validation_data_path.exists():
        validation_data_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        train_premise_embeddings.cpu(), train_data_path / "premise_embeddings.pt"
    )
    torch.save(
        train_hypothesis_embeddings.cpu(), train_data_path / "hypothesis_embeddings.pt"
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


def preprocess_corona(
    device: torch.device,
    data_path: Path = Path("data/processed/corona"),
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

    model_name: str = "all-mpnet-base-v2"

    # Load SentenceTransformer model
    model = load_mpnet_model(model_name=model_name, device=device)

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

    train_data_path = data_path / "train"
    validation_data_path = data_path / "validation"

    # Check if path exists
    if not train_data_path.exists():
        train_data_path.mkdir(parents=True, exist_ok=True)

    if not validation_data_path.exists():
        validation_data_path.mkdir(parents=True, exist_ok=True)

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


def extract_embeddings(
    sentences: List[str],
    model: SentenceTransformer,
    device: str = "cuda",
    save_path: Path = None,
) -> torch.Tensor:
    embeddings = model.encode(
        sentences,
        show_progress_bar=True,
        device=device,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    if save_path:
        # Check if path exists
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        # Save embeddings
        torch.save(embeddings, save_path)

    return embeddings


def load_mpnet_model(
    model_name: str = "all-mpnet-base-v2", device: str = "cuda"
) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    return model


def load_mpnet_tokenizer() -> transformers.AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    return tokenizer


def load_bert_model(
    model_name: str = "bert-base-uncased",
) -> transformers.BertPreTrainedModel:
    """
    Load the BERT model
    """
    bert_model = BertModel.from_pretrained(model_name)
    return bert_model


def load_bert_tokenizer(
    model_name: str = "bert-base-uncased",
) -> transformers.BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer
