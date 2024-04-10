from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
import torch

from src.utils.general import set_seed, configure_torch_backend
from src.data.utils import load_experiment_dataset
from src.utils.testing import TestClass
from src.models.predict_model import SimpleClassifier


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:
    # TODO: Fix, not finished
    seed: int = 42
    set_seed(seed)
    configure_torch_backend()

    # Define the model
    model = SimpleClassifier(in_dim=768, hidden_dims=[512, 256], out_dim=2)

    train_dataset, validation_dataset = load_experiment_dataset("sst2")

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    validation_data_loader = DataLoader(
        dataset=validation_dataset, batch_size=32, shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tester = TestClass(
        model=model,
        train_loader=train_data_loader,
        val_loader=validation_data_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    epochs: int = 20

    for epoch in range(epochs):
        train_loss, train_accuracy = tester.train_one_epoch()
        print(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}"
        )

        # Validation
        validation_loss, validation_accuracy = tester.test_function()
        print(
            f"Epoch: {epoch}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}"
        )
