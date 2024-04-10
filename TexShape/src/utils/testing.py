from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_lightning as pl

from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score


class TestClass:
    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.criterion: nn.Module = criterion
        self.optimizer: torch.optim.Optimizer = optimizer
        self.device: torch.device = device

    def test_function(self) -> None:
        raise NotImplementedError

    # Create a function to train for 1 epoch
    def train_one_epoch(self) -> Tuple[float, float]:
        self.model = self.model.to(self.device)
        self.model.train()

        all_preds = []
        all_labels = []
        preds: torch.Tensor
        labels: torch.Tensor
        loss: torch.Tensor
        inputs: torch.Tensor
        labels: torch.Tensor

        total_loss: float = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        accuracy = accuracy_score(all_labels, all_preds)

        return accuracy, total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float]:
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        all_preds = []
        all_labels = []
        labels: torch.Tensor
        preds: torch.Tensor
        loss: torch.Tensor
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        loss = self.criterion(outputs, labels)
        avg_loss = loss.item()

        return accuracy, avg_loss


####################### PyTorch Lightning Model #######################
class Model(pl.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.val_accuracy(y_hat, y)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_epoch_end(self, outputs):
        self.log("train_accuracy", self.train_accuracy.compute())

    def validation_epoch_end(self, outputs):
        self.log("val_accuracy", self.val_accuracy.compute())
