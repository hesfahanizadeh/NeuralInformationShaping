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
    def __init__(self, loss: str = "mine", **kwargs) -> None:
        super().__init__()
        self.energy_loss: Mine = kwargs.get("mine")
        self.kwargs: dict = kwargs
        self.gradient_batch_size: int = kwargs.get("gradient_batch_size", 1)
        self.train_loader = kwargs.get("train_loader")
        assert self.energy_loss is not None
        assert self.train_loader is not None
        print("energy loss: ", self.energy_loss)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        return self.energy_loss(x, z)

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=self.kwargs["lr"])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        # Each batch is just the minibatch in MINE, and also the same
        # as a "batch" from a dataloader, ie using our value of 'K'
        x, z = batch
        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        loss = self.energy_loss(x, z)
        mi = -loss
        
        
        tensorboard_logs = {"loss": loss, "mi": mi}
        tqdm_dict = {"loss_tqdm": loss, "mi": mi}
        self.last_mi = mi
        self.logger.experiment.add_scalar(
            f"MI Train | {self.kwargs['func']}, N={self.kwargs.get('N', 'conv mnist')}, batch_size={self.kwargs['batch_size']}, hidden=1",
            mi,
            self.current_epoch,
        )
        self.logger.log_metrics(tensorboard_logs, self.current_epoch)

        return {**tensorboard_logs, "log": tensorboard_logs, "progress_bar": tqdm_dict}
    
    def train_dataloader(self):
        return self.train_loader
    
def optimizer_step(
    self,
    epoch: int,
    batch_idx: int,
    optimizer: torch.optim.Optimizer,
    optimizer_idx: int = 0,
    optimizer_closure=None,
    on_tpu: bool = False,
    using_native_amp: bool = False,
    using_lbfgs: bool = False,
) -> None:
    if batch_idx % self.gradient_batch_size == 0:
        # Ensure closure is not None before attempting to use it
        if optimizer_closure is not None:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()  # Default step if no closure is provided
    else:
        # Optionally, you might decide to call the closure to update internal states or logs,
        # but this should be done carefully and documented why it's needed.
        if optimizer_closure is not None:
            optimizer_closure()
        # No step is performed here since it's not the right batch index for stepping.


    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int
    ) -> None:
        if batch_idx % self.gradient_batch_size == 0:
            optimizer.zero_grad()

    def train_dataloader(self):
        assert self.train_loader is not None
        return self.train_loader
