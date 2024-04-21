# Standard library imports
from pathlib import Path
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass
from abc import ABC

# Third-party library imports
import torch
import transformers
from transformers import BertTokenizer, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from omegaconf import DictConfig

class TexShapeDataset(torch.utils.data.Dataset):
    def __init__(
        self, embeddings: torch.Tensor, label1: torch.Tensor, label2: torch.Tensor
    ):
        self.embeddings = embeddings
        self.label1 = label1
        self.label2 = label2

    def switch_labels(self) -> None:
        self.label1, self.label2 = self.label2, self.label1

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.label1[idx], self.label2[idx]


class MNLI_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        premise: torch.Tensor,
        hypothesis: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor,
    ):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label1 = label1
        self.label2 = label2

    def __len__(self) -> int:
        return len(self.premise)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.premise[idx],
            self.hypothesis[idx],
            self.label1[idx],
            self.label2[idx],
        )


class DatasetParams(ABC):
    dataset_loc: Union[Path, str]
    dataset_name: str


@dataclass
class SST2_Params(DatasetParams):
    dataset_name: str
    sent_len_threshold: int
    train_test_split_ratio: float
    dataset_loc: Path

    def __post_init__(self):
        if isinstance(self.dataset_loc, str):
            self.dataset_loc = Path(self.dataset_loc)


@dataclass
class MNLI_Params(DatasetParams):
    dataset_name: str
    dataset_loc: Path
    combination_type: str

    def __post_init__(self):
        if isinstance(self.dataset_loc, str):
            self.dataset_loc = Path(self.dataset_loc)


def load_experiment_dataset(
    *,
    dataset_params: DatasetParams,
    device: torch.device,
) -> TexShapeDataset:
    data_loc = dataset_params.dataset_loc
    dataset_name = dataset_params.dataset_name

    if not data_loc.exists():
        data_loc.mkdir(parents=True, exist_ok=True)

    if dataset_name == "sst2":
        sst2_dataset_params: SST2_Params = dataset_params
        # Check if path exists
        dataset_path: Path = data_loc / "dataset.pt"

        if not dataset_path.exists():
            preprocess_sst2(device=device, data_path=data_loc)

        train_test_split_ratio: str = sst2_dataset_params.train_test_split_ratio
        train_dataset, _ = load_sst2(
            sst2_data_path=data_loc, train_test_split_ratio=train_test_split_ratio
        )

    elif dataset_name == "mnli":
        train_dataset: MNLI_Dataset
        mnli_dataset_params: MNLI_Params = dataset_params

        data_loc = data_loc / "mnli"
        train_dataset, _ = load_mnli(data_path=data_loc)
        train_premise = train_dataset.premise
        train_hypothesis = train_dataset.hypothesis
        train_label1 = train_dataset.label1
        train_label2 = train_dataset.label2

        combination_type = mnli_dataset_params.combination_type
        if combination_type == "concat":
            train_embeddings = torch.cat((train_premise, train_hypothesis), dim=-1)

        elif combination_type == "join":
            train_embeddings = torch.cat((train_premise, train_hypothesis), dim=0)
            train_label2 = torch.cat((train_label2, train_label2), dim=0)

        elif combination_type == "premise_only":
            train_embeddings = train_premise

        train_dataset = TexShapeDataset(train_embeddings, train_label1, train_label2)

    elif dataset_name == "corona":
        data_loc = data_loc / "corona"
        train_dataset, _ = load_corona(data_path=data_loc)
    else:
        raise ValueError("Invalid dataset")

    return train_dataset


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

def get_dataset_params(dataset_name: str, config: DictConfig) -> DatasetParams:
    dataset_params = config.dataset
    if dataset_name == "sst2":
        return SST2_Params(**dataset_params)
    raise ValueError("Invalid dataset name")

def preprocess_sst2(
    device: torch.Tensor, data_path: Path
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    model_name: str = "all-mpnet-base-v2"
    sent_len_threshold: int = 8

    tokenizer = load_mpnet_tokenizer()
    model = load_mpnet_model(model_name=model_name, device=device)

    dataset = load_dataset("stanfordnlp/sst2")
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

    sentiment_labels = torch.tensor(dataset["label"])
    sent_len_labels = torch.tensor(dataset["sent_len"])

    dataset = TexShapeDataset(embeddings, sentiment_labels, sent_len_labels)

    # Check if path exists
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, data_path / "dataset.pt")
    return dataset


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


def load_sst2(
    sst2_data_path: Path, train_test_split_ratio: float = 0.9
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    """
    Load the SST-2 dataset and split it into train and validation sets.

    :param sst2_data_path: The path to the SST-2 dataset.
    :param train_test_split_ratio: The ratio to split the dataset into train and validation sets. Default is 0.9.

    :return: A tuple containing the train and validation datasets.
    """
    dataset: TexShapeDataset = torch.load(sst2_data_path / "dataset.pt")

    # Train Test Split
    train_size = int(train_test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataset = train_dataset.dataset
    validation_dataset = validation_dataset.dataset
    return train_dataset, validation_dataset


def load_mnli(
    mnli_data_path: Path = Path("data/processed/mnli"),
) -> Tuple[MNLI_Dataset, MNLI_Dataset]:
    train_dataset: MNLI_Dataset = torch.load(mnli_data_path / "train.pt")
    validation_dataset: MNLI_Dataset = torch.load(mnli_data_path / "validation.pt")
    return train_dataset, validation_dataset


def load_corona(
    data_path: Path = Path("data/processed/corona"),
) -> Tuple[TexShapeDataset, TexShapeDataset]:
    train_dataset: TexShapeDataset = torch.load(data_path / "train.pt")
    validation_dataset: TexShapeDataset = torch.load(data_path / "validation.pt")
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
