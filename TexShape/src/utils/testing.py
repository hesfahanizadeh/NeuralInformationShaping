from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
# Import functional as f
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from torchmetrics import Accuracy
import torchmetrics
from sklearn.metrics import accuracy_score


# class TestClass(pl.LightningModule):
#     def __init__(self, model, lr=1e-3):
#         super().__init__()
#         self.model = model
#         self.lr = lr
#         self.criterion: nn.Module = nn.CrossEntropyLoss()
#         self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
#         self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        
#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         # it is independent of forward
#         x, y = batch
#         y_hat = self.model(x)
#         loss = nn.functional.cross_entropy(y_hat, y)
        
#         acc = self.train_acc.compute()  # Computes the current accuracy
#         self.log('train_acc_step', acc, prog_bar=True)
#         self.train_acc.reset() 
#         return loss
    
#     def on_train_epoch_end(self):
#         self.log('train_acc_epoch', self.train_acc, prog_bar=True)
        
#     def validation_step(self, batch, batch_idx):
#         x, y = batch 
#         y_hat = self.forward(x) 
#         loss = self.criterion(y_hat, y) 
#         self.valid_acc(y_hat, y)
#         self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
#         return loss
           
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer

class TestClass(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.accuracy_train = Accuracy(task="multiclass", num_classes=2)
        self.accuracy_val = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy_train.update(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.accuracy_train.compute(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # On the end of the epoch, we want to reset the accuracy
    def on_training_epoch_end(self):
        self.accuracy_train.reset()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy_val.update(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy_val.compute(), prog_bar=True)
        return loss

    # On the end of the epoch, we want to reset the accuracy
    def on_validation_epoch_end(self):
        self.accuracy_val.reset()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.002)
# class TestClass:
#     def __init__(
#         self,
#         *,
#         model: nn.Module,
#         train_loader: DataLoader,
#         val_loader: DataLoader,
#         criterion: nn.Module,
#         optimizer: torch.optim.Optimizer,
#         device: torch.device,
#     ) -> None:
#         self.model: nn.Module = model
#         self.train_loader: DataLoader = train_loader
#         self.val_loader: DataLoader = val_loader
#         self.criterion: nn.Module = criterion
#         self.optimizer: torch.optim.Optimizer = optimizer
#         self.device: torch.device = device

#     def test_function(self) -> None:
#         raise NotImplementedError

#     # Create a function to train for 1 epoch
#     def train_one_epoch(self) -> Tuple[float, float]:
#         self.model = self.model.to(self.device)
#         self.model.train()

#         all_preds = []
#         all_labels = []
#         preds: torch.Tensor
#         labels: torch.Tensor
#         loss: torch.Tensor
#         inputs: torch.Tensor
#         labels: torch.Tensor

#         total_loss: float = 0.0
#         for inputs, labels in self.train_loader:
#             inputs, labels = inputs.to(self.device), labels.to(self.device)

#             self.optimizer.zero_grad()

#             outputs = self.model(inputs)
#             _, preds = torch.max(outputs, 1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#             loss = self.criterion(outputs, labels)
#             loss.backward()
#             self.optimizer.step()

#             total_loss += loss.item()

#         accuracy = accuracy_score(all_labels, all_preds)

#         return accuracy, total_loss / len(self.train_loader)

#     def validate(self) -> Tuple[float, float]:
#         self.model = self.model.to(self.device)
#         self.model = self.model.eval()
#         all_preds = []
#         all_labels = []
#         labels: torch.Tensor
#         preds: torch.Tensor
#         loss: torch.Tensor
#         with torch.no_grad():
#             for inputs, labels in self.val_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)

#                 outputs = self.model(inputs)
#                 _, preds = torch.max(outputs, 1)

#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())

#         accuracy = accuracy_score(all_labels, all_preds)
#         loss = self.criterion(outputs, labels)
#         avg_loss = loss.item()

#         return accuracy, avg_loss


# ####################### PyTorch Lightning Model #######################
# class Model(pl.LightningModule):
#     def __init__(self, model, criterion):
#         super().__init__()
#         self.model = model
#         self.criterion = criterion
#         self.train_accuracy = Accuracy()
#         self.val_accuracy = Accuracy()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         self.log("val_loss", loss)
#         self.val_accuracy(y_hat, y)

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_epoch_end(self, outputs):
#         self.log("train_accuracy", self.train_accuracy.compute())

#     def validation_epoch_end(self, outputs):
#         self.log("val_accuracy", self.val_accuracy.compute())

# def train_and_validate_model(
#     model, train_data_loader, validation_data_loader, device, **kwargs
# ):
#     """Train and validate the model."""
#     lr = kwargs.get("lr", 0.001)
#     # Define the loss function and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     tester = TestClass(
#         model=model,
#         train_loader=train_data_loader,
#         val_loader=validation_data_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         device=device,
#     )

#     epochs: int = 20

#     for epoch in range(epochs):
#         train_loss, train_accuracy = tester.train_one_epoch()
#         logging.info(
#             "Epoch: %s, Train Loss: %s, Train Accuracy: %s",
#             epoch,
#             train_loss,
#             train_accuracy,
#         )

#         # Validation
#         validation_loss, validation_accuracy = tester.train_one_epoch()
#         logging.info(
#             "Epoch: %s, Validation Loss: %s, Validation Accuracy: %s",
#             epoch,
#             validation_loss,
#             validation_accuracy,
#         )
