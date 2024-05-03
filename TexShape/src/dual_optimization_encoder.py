import math
from typing import Tuple
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from src.models import models_to_train
from src.models.models_to_train import MI_CalculatorModel
from src.models.utils import create_mi_calculator_model
from src.utils.config import (
    ExperimentParams,
    MineParams,
    EncoderParams,
)
from src.mine import MutualInformationEstimator, Mine
from src.data.utils import TexShapeDataset


class DualOptimizationEncoder(nn.Module):
    """
    Dual Optimization Encoder Model. This model trains the encoder using dual optimization.
    """

    def __init__(
        self,
        *,
        experiment_params: ExperimentParams,
        encoder_model: models_to_train.Encoder,
        data_loader: DataLoader,
        device: torch.device,
        experiment_dir_path: Path,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder_model = encoder_model

        self.data_loader: DataLoader = data_loader
        self.dataset: TexShapeDataset = data_loader.dataset

        self.experiment_params: ExperimentParams = experiment_params
        self.mine_params: MineParams = experiment_params.mine_params

        # Set the mine batch size if -1 passed
        if self.mine_params.mine_batch_size == -1:
            self.mine_params.mine_batch_size = len(self.dataset)

        self.encoder_params: EncoderParams = experiment_params.encoder_params
        self.experiment_dir_path: Path = experiment_dir_path

        self.encoder_learning_rate = (
            experiment_params.encoder_params.encoder_learning_rate
        )

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder_model.parameters(),
            lr=self.encoder_learning_rate,
        )
        self.beta: float = experiment_params.beta
        # Define the device
        self.device: torch.device = device

        if kwargs.get("device_idx", None) is not None:
            self.device_idx: int = kwargs.get("device_idx", None)
        else:
            self.device_idx: int = 0

        self.num_workers: int = 0  # experiment_params.num_workers

        logging.info("Device: %s", self.device)

        self.epoch = 0

    def get_MINE(
        self,
        *,
        stats_network: nn.Module,
        device: torch.device,
    ) -> MutualInformationEstimator:
        # Define Mine model
        mi_estimator = Mine(stats_network, loss="mine").to(self.device)

        kwargs = {
            "mine": mi_estimator,
            "lr": 1e-3,
            "alpha": 0.1,  # Used as the ema weight in MINE
            # Determines how many mini batches (MINE iters) of gradients get accumulated before optimizer step gets applied
            # Meant to stabilize the MINE curve for [hopefully] better encoder training performance
        }

        model = MutualInformationEstimator(
            loss="mine",
            **kwargs,
        ).to(device)
        return model

    def get_transformed_data_and_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
        # Get encoder transformed data
        transformed_embeddings: torch.Tensor = self.encoder_model(
            self.dataset.embeddings.float().to(self.device)
        )

        labels_public = self.dataset.label1.float()
        labels_private = self.dataset.label2.float()

        # Define datasets for MINE

        # Public Label
        z_train_utility_detached = TensorDataset(
            transformed_embeddings.detach().cpu(),
            labels_public.detach(),
        )

        # Private Label
        z_train_privacy_detached = TensorDataset(
            transformed_embeddings.detach().cpu(),
            labels_private.detach(),
        )

        # Define dataloaders for MINE
        z_train_loader_utility_detached = DataLoader(
            z_train_utility_detached,
            self.mine_params.mine_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # TODO: Check here
            # pin_memory=True,
        )
        z_train_loader_privacy_detached = DataLoader(
            z_train_privacy_detached,
            self.mine_params.mine_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # TODO: Check here
            # pin_memory=True,
        )

        return (
            z_train_loader_utility_detached,
            z_train_loader_privacy_detached,
            transformed_embeddings,
        )

    def forward(
        self,
        *,
        num_batches_final_MI: int,
        include_privacy: bool = True,
        include_utility: bool = True,
        gradient_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the dual optimization model

        Args:
            num_batches_final_MI (int): Number of batches to calculate the final MI estimate
            include_privacy (bool, optional): Include privacy in the training. Defaults to True.
            include_utility (bool, optional): Include utility in the training. Defaults to True.
            gradient_batch_size (int, optional): . Defaults to 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: MI estimates for utility and privacy
        """

        transformed_embeddings: torch.Tensor

        (
            z_train_loader_utility_detached,
            z_train_loader_privacy_detached,
            transformed_embeddings,
        ) = self.get_transformed_data_and_loaders()

        # TODO: Fix this part
        # Get MINE model (sitting in Pytorch lightning module)
        model_MINE_utility: MutualInformationEstimator
        model_MINE_privacy: MutualInformationEstimator

        MINE_utility_stats_network: nn.Module = self._get_utility_stats_network()
        MINE_privacy_stats_network: nn.Module = self._get_privacy_stats_network()

        model_MINE_utility = self.get_MINE(
            stats_network=MINE_utility_stats_network,
            device=self.device,
        )

        model_MINE_privacy = self.get_MINE(
            stats_network=MINE_privacy_stats_network,
            device=self.device,
        )

        # TODO: Fix this part
        # # If previous MINE model exists, load it
        # if self.utility_stats_network_path:
        #     try:
        #         model_MINE_utility.energy_loss.load_state_dict(
        #             torch.load(self.utility_stats_network_path)
        #         )
        #     except FileNotFoundError:
        #         logging.info("No previous MINE model found, training from scratch")

        # if self.privacy_stats_network_path:
        #     try:
        #         model_MINE_privacy.energy_loss.load_state_dict(
        #             torch.load(self.privacy_stats_network_path)
        #         )
        #     except FileNotFoundError:
        #         logging.info("No previous MINE model found, training from scratch")

        # Optimize MINE estimate, "train" MINE
        # pylint: disable=no-member
        last_mi_utility = torch.tensor(0)
        last_mi_privacy = torch.tensor(0)
        if include_utility:
            early_stop_callback = EarlyStopping(
                monitor="mi",
                patience=self.mine_params.mine_trainer_patience,
                mode="max",
            )

            logger_utility = TensorBoardLogger(
                str(self.experiment_dir_path),
                name="MINE_logs",
                version=f"utility_{self.epoch}",
            )

            trainer_utility = Trainer(
                max_epochs=self.mine_params.mine_epochs_utility,
                logger=logger_utility,
                log_every_n_steps=1,
                accelerator="gpu",
                devices=[self.device_idx],
                accumulate_grad_batches=gradient_batch_size,
                callbacks=[early_stop_callback],
            )

            trainer_utility.fit(
                model=model_MINE_utility,
                train_dataloaders=z_train_loader_utility_detached,
            )

            # TODO: Fix this part
            # if self.utility_stats_network_path:
            #     logging.info
            # ("Using MI Strategy, saving MINE model")
            #     # Save the weights of the MINE model
            #     torch.save(
            #         model_MINE_utility.energy_loss.state_dict(),
            #         self.utility_stats_network_path,
            #     )

            ## -------- Calculate I(T(x); L(x)) estimate after MINE training ---------- ##
            # **IMPORTANT**: Use the non-detached og transformed_embeddings so that gradients are retained

            labels_public = self.dataset.label1.float()
            z_train_utility = TensorDataset(
                transformed_embeddings, labels_public.float()
            )

            # z_train_utility = CustomDataset(
            #     transformed_embeddings, self.data_loader.dataset.inputs.float().to(device)
            # )

            z_train_loader_utility = DataLoader(
                z_train_utility,
                batch_size=self.mine_params.mine_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            model_MINE_utility.energy_loss.to(self.device)
            sum_MI_utility = 0

            # Average MI across num_batches_final_MI batches to lower variance
            # Batches are K random samples from the dataset after all

            logging.info(
                "Num batches final MI: %s, \tLen dataset: %s, \tK: %s, \tLen dataset / K: %s",
                num_batches_final_MI,
                len(z_train_loader_utility.dataset),
                self.mine_params.mine_batch_size,
                len(z_train_loader_utility.dataset) / self.mine_params.mine_batch_size,
            )

            assert num_batches_final_MI <= (
                math.ceil(
                    len(z_train_loader_utility.dataset)
                    / self.mine_params.mine_batch_size
                )
            )
            utility_it = iter(z_train_loader_utility)
            for _ in range(num_batches_final_MI):
                Tx, Lx = next(utility_it)
                Tx, Lx = Tx.to(self.device), Lx.to(self.device)
                sum_MI_utility += model_MINE_utility.energy_loss(Tx, Lx)

            # MINE loss = -1 * MI estimate since we are maximizing using gradient descent still
            last_mi_utility: torch.Tensor = -1 * sum_MI_utility / num_batches_final_MI

        if include_privacy:
            early_stop_callback = EarlyStopping(
                monitor="mi",
                patience=self.mine_params.mine_trainer_patience,
                mode="max",
            )

            logger_privacy = TensorBoardLogger(
                str(self.experiment_dir_path),
                name="MINE_logs",
                version=f"privacy_{self.epoch}",
            )

            trainer = Trainer(
                max_epochs=self.mine_params.mine_epochs_privacy,
                logger=logger_privacy,
                log_every_n_steps=1,
                accelerator="gpu",
                devices=[self.device_idx],
                accumulate_grad_batches=gradient_batch_size,
                callbacks=[early_stop_callback],
            )

            trainer.fit(
                model=model_MINE_privacy,
                train_dataloaders=z_train_loader_privacy_detached,
            )

            # # If path exists, save the weights of the MINE model
            # if self.privacy_stats_network_path:
            #     logging.info("Using MI Strategy, saving MINE model")
            #     torch.save(
            #         model_MINE_privacy.energy_loss.state_dict(),
            #         self.privacy_stats_network_path,
            #     )

            #     ## -------- Calculate I(T(x); S(x)) estimate after MINE training ---------- ##
            labels_private = self.dataset.label2.float()
            z_train_privacy = TensorDataset(transformed_embeddings, labels_private)
            z_train_loader_privacy = DataLoader(
                z_train_privacy,
                self.mine_params.mine_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            model_MINE_privacy.energy_loss.to(self.device)

            assert num_batches_final_MI <= math.ceil(
                len(z_train_loader_privacy.dataset) / self.mine_params.mine_batch_size
            )
            sum_MI_privacy = 0
            privacy_it = iter(z_train_loader_privacy)
            # prev_mi = None
            for _ in range(num_batches_final_MI):
                Tx: torch.Tensor
                Sx: torch.Tensor
                Tx, Sx = next(privacy_it)
                Tx, Sx = Tx.to(self.device), Sx.to(self.device)
                sum_MI_privacy += model_MINE_privacy.energy_loss(Tx, Sx)

            last_mi_privacy: torch.Tensor = -1 * sum_MI_privacy / num_batches_final_MI

        logging.info(
            "Final MI values: utility: %s, privacy: %s",
            last_mi_utility.item(),
            last_mi_privacy.item(),
        )
        return last_mi_utility, last_mi_privacy

    def train_encoder(
        self,
        *,
        num_batches_final_MI: int = 100,
        include_privacy: bool = True,
        include_utility: bool = True,
        gradient_batch_size: int = 1,
    ) -> None:
        """
        Training function for the encoder model.

        Args:
            num_batches_final_MI (int, optional): Number of batches to calculate the final MI estimate. Defaults to 100.
            include_privacy (bool, optional): Include privacy in the training. Defaults to True.
            include_utility (bool, optional): Include utility in the training. Defaults to True.
            gradient_batch_size (int, optional): . Defaults to 1.

        Returns:
            None
        """
        """K = MINE_BATCH_SIZE"""
        # Encoder's training params
        mi_utility: torch.Tensor
        mi_privacy: torch.Tensor
        self.encoder_model.train()

        for epoch in range(self.encoder_params.num_enc_epochs):
            (
                mi_utility,
                mi_privacy,
            ) = self.forward(
                num_batches_final_MI=num_batches_final_MI,
                include_privacy=include_privacy,
                include_utility=include_utility,
                gradient_batch_size=gradient_batch_size,
            )
            self.encoder_optimizer.zero_grad()

            # Calculate the score
            loss: torch.Tensor = -mi_utility + self.beta * mi_privacy
            loss.backward()
            self.encoder_optimizer.step()

            # # Save the scores
            # self.utility_scores.append(mi_utility.detach().cpu())

            # if include_privacy:
            #     self.privacy_scores.append(mi_privacy.detach().cpu())
            enc_save_dir = self.experiment_dir_path / "encoder_weights"

            if not enc_save_dir.exists():
                enc_save_dir.mkdir(parents=True, exist_ok=True)

            enc_save_path = enc_save_dir / f"model_{epoch}.pt"
            self._save_encoder_weights(enc_save_path)

            logging.info(
                "====> Epoch: %s Utility MI I(T(x); L(x)): %s",
                epoch,
                round(mi_utility.item(), 8),
            )
            logging.info(
                "====> Epoch: %s Privacy MI I(T(x); S(x)): %s",
                epoch,
                round(mi_privacy.item(), 8),
            )
            logging.info("====> Epoch: %s Loss: %s", epoch, round(loss.item(), 8))

            self.epoch += 1

    def _get_utility_stats_network(self) -> MI_CalculatorModel:
        stats_network = create_mi_calculator_model(
            model_name=self.mine_params.utility_stats_network_model.model_name,
            model_params=self.mine_params.utility_stats_network_model.model_params,
        )
        return stats_network

    def _get_privacy_stats_network(self) -> MI_CalculatorModel:
        stats_network = create_mi_calculator_model(
            model_name=self.mine_params.privacy_stats_network_model.model_name,
            model_params=self.mine_params.privacy_stats_network_model.model_params,
        )
        return stats_network

    def _save_encoder_weights(self, model_path: Path) -> None:
        # Don't save the state dict since that doesn't include the model parameters + their gradients
        # Options were to save entire model or optimizer's state dict:
        # https://discuss.pytorch.org/t/how-to-save-the-requires-grad-state-of-the-weights/52906/6

        # Get the name of the parent dir of the model_path
        save_dir = model_path.parent

        logging.debug("Saving weights to %s", save_dir)

        # Save the state dict of the model
        torch.save(
            self.encoder_model.state_dict(),
            model_path,
        )

        # optimizer_path = (
        #     enc_save_dir_path
        #     / f"[optimizer] {self.experiment_params.experiment_name}-epoch={epoch}.pt"
        # )
        # torch.save(
        #     self.encoder_optimizer.state_dict(),
        #     optimizer_path,
        # )

    def save_mi_scores(self, epoch: int) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    logging.info("Test")
