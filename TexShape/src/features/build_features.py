import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer
import transformers
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path
from ..data.utils import TexShapeDataset, MNLI_Dataset


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


def preprocess_sst2(
    device="cuda", data_path=Path("data/processed/sst2")
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    model_name: str = "all-mpnet-base-v2"
    sent_len_threshold: int = 8
    train_test_split_ratio: float = 0.9

    tokenizer = load_mpnet_tokenizer()
    model = load_mpnet_model(model_name=model_name, device=device)

    dataset = load_dataset("glue", "sst2")
    dataset = dataset["train"]

    embeddings: torch.Tensor = extract_embeddings(
        dataset["sentence"], model, device=device
    )

    def get_sent_len(example):
        tokenized_sentence = tokenizer.tokenize(example["sentence"])
        sentence_length = len(tokenized_sentence)
        # Label sentence_length
        if sentence_length <= sent_len_threshold:
            sent_len_label = 0
        sent_len_label = 1
        return {"sent_len": sent_len_label}

    dataset = dataset.map(get_sent_len)

    # Save dataset
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    sentiment_labels = torch.tensor(dataset["label"])
    sent_len_labels = torch.tensor(dataset["sent_len"])

    dataset = TexShapeDataset(embeddings, sentiment_labels, sent_len_labels)

    # Train Test Split
    train_size = int(train_test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Check if path exists
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    # Save dataset
    torch.save(train_dataset, data_path / "train.pt")
    torch.save(test_dataset, data_path / "validation.pt")
    return train_dataset, test_dataset


def preprocess_mnli(
    device: str,
    data_path: Path = Path("data/processed/mnli"),
) -> Tuple[MNLI_Dataset, MNLI_Dataset]:
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

    torch.save(train_dataset, data_path / "train.pt")
    torch.save(validation_dataset, data_path / "validation.pt")
    return train_dataset, validation_dataset

def preprocess_corona(
    device: str,
    data_path: Path = Path("data/processed/corona"),
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    
    """Load the Corona dataset
    TODO: Fix this function
    Args:
        train_data_path (Path): Path object to the training data
        validation_data_path (Path): Path object to the validation data

    Returns:
        Tuple[ Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ]:
        Tuple containing the training and validation data
    """
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

    train_dataset = TexShapeDataset(
        train_inputs, train_targets_public, train_targets_private
    )
    validation_dataset = TexShapeDataset(
        validation_inputs, validation_targets_public, validation_targets_private
    )
    return train_dataset, validation_dataset

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


if __name__ == "__main__":
    pass
