import torch

# Import functional as f
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import Accuracy


# pylint: disable=no-member, unused-argument
class TestClass(pl.LightningModule):
    def __init__(self, model, num_class):
        super().__init__()
        self.model = model
        self.accuracy_train = Accuracy(task="multiclass", num_classes=num_class)
        self.accuracy_val = Accuracy(task="multiclass", num_classes=num_class)

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy_train.update(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.accuracy_train,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy_val.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy_val, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
