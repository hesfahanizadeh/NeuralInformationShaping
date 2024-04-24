from pathlib import Path
from typing import List, Dict

import torch
import transformers
from transformers import BertTokenizer, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


def preprocess_sst2(device: torch.Tensor, data_path: Path) -> None:
    model_name: str = "all-mpnet-base-v2"
    sent_len_threshold: int = 8

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
        if sentence_length <= sent_len_threshold:
            sent_len_label = 0
        sent_len_label = 1
        return {"sent_len": sent_len_label}

    dataset = dataset.map(get_sent_len)

    sentiment_labels = torch.tensor(dataset["label"]).detach().cpu()
    sent_len_labels = torch.tensor(dataset["sent_len"]).detach().cpu()

    # Check if path exists
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    torch.save(embeddings, data_path / "embeddings.pt")
    torch.save(sentiment_labels, data_path / "sentiment_labels.pt")
    torch.save(sent_len_labels, data_path / "sent_len_labels.pt")


def preprocess_mnli(
    device: str,
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

    train_label1 = torch.tensor(train_filtered_dataset["label"])
    train_label2 = train_filtered_dataset["genre"]
    train_label2 = torch.tensor(
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

    validation_label1 = torch.tensor(validation_filtered_dataset["label"])

    # Obtain private targets
    validation_label2 = validation_filtered_dataset["genre"]
    validation_label2 = torch.tensor(
        [0 if label == "government" else 1 for label in validation_label2]
    )

    validation_premise_embeddings = extract_embeddings(
        validation_filtered_dataset["premise"], model, device=device
    )

    validation_hypothesis_embeddings = extract_embeddings(
        validation_filtered_dataset["hypothesis"], model, device=device
    )

    torch.save(train_premise_embeddings, data_path / "train_premise_embeddings.pt")
    torch.save(
        train_hypothesis_embeddings, data_path / "train_hypothesis_embeddings.pt"
    )
    torch.save(train_label1, data_path / "train_label1.pt")
    torch.save(train_label2, data_path / "train_label2.pt")

    torch.save(
        validation_premise_embeddings, data_path / "validation_premise_embeddings.pt"
    )
    torch.save(
        validation_hypothesis_embeddings,
        data_path / "validation_hypothesis_embeddings.pt",
    )
    torch.save(validation_label1, data_path / "validation_label1.pt")
    torch.save(validation_label2, data_path / "validation_label2.pt")


def preprocess_corona(
    device: str,
    data_path: Path = Path("data/processed/corona"),
) -> None:
    """Load the Corona dataset
    TODO: Fix this function
    Args:
        train_data_path (Path): Path object to the training data
        validation_data_path (Path): Path object to the validation data

    Returns:
        Tuple[ Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ]:
        Tuple containing the training and validation data
    """
    raise NotImplementedError("Not implemented yet.")
    # Load the training data
    # TODO: Fix type hinting
    train_data_path = data_path / "train.pt"
    validation_data_path = data_path / "validation.pt"
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
