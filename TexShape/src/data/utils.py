from pathlib import Path
from typing import Tuple, List

import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer, AutoTokenizer

from src.utils.config import ExperimentType


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


def load_mpnet_model_tokenizer(
    model_name: str, device: torch.device
) -> Tuple[SentenceTransformer, AutoTokenizer]:
    model = SentenceTransformer(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    return model, tokenizer


def load_bert_model_tokenizer(
    model_name: str = "bert-base-uncased",
) -> transformers.BertPreTrainedModel:
    """
    Load the BERT model
    """
    bert_model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return bert_model, tokenizer


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
