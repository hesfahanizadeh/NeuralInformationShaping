import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AutoTokenizer
import transformers
from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from abc import ABC, abstractmethod
from pathlib import Path

@dataclass
class ProcessedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor  # TODO: Check this


@dataclass
class EmbeddingExtractorOutput:
    embeddings: List[torch.Tensor]
    labels: List[torch.Tensor]


class EmbeddingExtractor(ABC):
    @abstractmethod
    def extract_embeddings():
        raise NotImplementedError

    @abstractmethod
    def save_embeddings():
        raise NotImplementedError


class SST2EmbeddingExtractor(EmbeddingExtractor):
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: transformers.AutoTokenizer,
        device: str = None,
        sent_len_threshold: int = 8
    ) -> None:

        self.device: str = device
        self.tokenizer: transformers.AutoTokenizer = tokenizer
        self.model: SentenceTransformer = model.to(self.device)
        self.model.eval()
        dataset = load_dataset("glue", "sst2")

        # Extract train and val splits
        train_dataset = dataset["train"]
        # Train test split the train split
        train_test_split = train_dataset.train_test_split(test_size=0.1)

        # Extract train and val splits from train_test_split
        self.train_dataset = train_test_split["train"]
        self.test_dataset = train_test_split["test"]

        self.sent_len_threshold: int = sent_len_threshold

    def extract_embeddings(self) -> None:
        # Convert embeddings and labels to PyTorch tensors
        self.train_embeddings = self.model.encode(
            self.train_dataset["sentence"],
            show_progress_bar=True,
            device=self.device,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self.train_sentiment_labels = torch.tensor(self.train_dataset["label"])
        self.train_sent_len_labels = torch.tensor(self.train_dataset["sent_len"])

        self.val_embeddings = self.model.encode(
            self.test_dataset["sentence"],
            show_progress_bar=True,
            device="cuda",
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self.val_sentiment_labels = torch.tensor(self.test_dataset["label"])
        self.val_sent_len_labels = torch.tensor(self.test_dataset["sent_len"])

    def save_embeddings(self, data_path = Path("data/processed/sst2")) -> None:
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)

        train_embeddings_save_path = data_path / "train" / "embeddings.pt"
        train_sentiment_labels_save_path = data_path / "train" / "sentiment_labels.pt"
        train_sent_len_labels_save_path = data_path / "train" / "sent_len_labels.pt"

        val_embeddings_save_path = data_path / "validation" / "embeddings.pt"
        val_sentimnet_labels_save_path = data_path / "validation" / "sentiment_labels.pt"
        val_sent_len_labels_save_path = data_path / "validation" / "sent_len_labels.pt"

        torch.save(self.train_embeddings, train_embeddings_save_path)
        torch.save(self.train_sentiment_labels, train_sentiment_labels_save_path)
        torch.save(self.train_sent_len_labels, train_sent_len_labels_save_path)

        torch.save(self.val_embeddings, val_embeddings_save_path)
        torch.save(self.val_sentiment_labels, val_sentimnet_labels_save_path)
        torch.save(self.val_sent_len_labels, val_sent_len_labels_save_path)

    def is_greater_than(self, *, sentence_length: int) -> int:
        # Label sentence_length
        if sentence_length <= self.sent_len_threshold:
            return 0
        return 1

    def _get_sent_len_label(self, sample) -> dict:
        # Tokenize sentence
        tokenized_sentence = tokenizer.tokenize(sample["sentence"])
        # Get sentence length
        sentence_length: int = len(tokenized_sentence)
        # Assign label
        label: int = self.is_greater_than(sentence_length=sentence_length)
        return {"sent_len": label}

    def add_sent_len_labels(self) -> None:
        self.train_dataset.map(self._get_sent_len_label)


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


def load_mpnet_model(
    model_name: str = "all-mpnet-base-v2", device: str = "cuda"
) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    return model


def load_mpnet_tokenizer() -> transformers.AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    return tokenizer


if __name__ == "__main__":
    tokenizer = load_mpnet_tokenizer()
    model = load_mpnet_model()
    # Load sst2 dataset from huggingface datasets library
    sst2_embedding_extractor = SST2EmbeddingExtractor(model=model, tokenizer=tokenizer, device="cpu")
