import math
from typing import Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

torch.autograd.set_detect_anomaly(True)
EPS = 1e-6

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input: torch.Tensor
        running_mean: torch.Tensor
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output * input.exp().detach() / (running_mean + EPS) / input.shape[0]
        )
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x: torch.Tensor, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(
        self,
        stats_network: nn.Module,
        loss: str = "mine",
        alpha: float = 0.01,
        lam: float = 0.1,
        C=0,
    ) -> None:
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha  # Used for ema during MINE iterations
        # Both lambda and C are a part of the regularization in ReMINE's objective
        self.lam = lam
        self.C = C
        self.stats_network: nn.Module = stats_network

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, z_marg: torch.Tensor = None
    ) -> torch.Tensor:
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        stats_network_output: torch.Tensor = self.stats_network(x, z)
        stats_network_score: torch.Tensor = stats_network_output.mean()
        t_marg: torch.Tensor = self.stats_network(x, z_marg)

        if self.loss in ["mine"]:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha
            )
        elif self.loss in ["fdiv"]:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ["mine_biased"]:
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])

        # Introducing ReMINE regularization here
        return (
            -stats_network_score + second_term + self.lam * (second_term - self.C) ** 2
        )


class MutualInformationEstimator(pl.LightningModule):
    def __init__(
        self, loss: str = "mine", **kwargs
    ) -> None:
        super().__init__()
        self.energy_loss: Mine = kwargs.get("mine")
        self.kwargs = kwargs
        assert self.energy_loss is not None
        print("energy loss: ", self.energy_loss)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        return self.energy_loss(x, z)

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=self.kwargs["lr"])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        x, z = batch
        
        # Ensure the data is on the correct device
        x = x.to(self.device)
        z = z.to(self.device)

        loss = self.energy_loss(x, z)
        mi = -loss

        # Log the metrics using self.log
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mi', mi, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # # Maintain some state if needed
        # self.last_mi = mi

        # Return the loss to use the default backward pass
        return loss
