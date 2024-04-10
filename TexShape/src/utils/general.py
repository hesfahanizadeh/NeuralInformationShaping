from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from pytorch_lightning import seed_everything
import numpy as np

# ROC and AUC functions # TODO Optimize this function
def get_roc_auc(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model = model.to(device)
    model = model.eval()

    # Get the predictions and labels
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(outputs.cpu().numpy()[:, 1])

    roc = roc_curve(all_labels, all_confidences)

    # Get the AUC
    auc_score = auc(roc[0], roc[1])
    return roc, auc_score

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed)

def configure_torch_backend():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_include_privacy(experiment_type: str) -> bool:
    if experiment_type == "utility+privacy" or experiment_type == "compression+filter":
        return True
    return False