import math
from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models import models_to_train
from utils.data_structures import ExperimentParams, MINE_Params
from mine import MutualInformationEstimator, Mine
from data.utils import CustomDataset

class DualOptimizationEncoder(nn.Module):
    def __init__(
        self,
        *,
        experiment_params: ExperimentParams,
        encoder_model: models_to_train.Encoder,
        data_loader: DataLoader,
        device: torch.device,
        private_labels: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.encoder_model = encoder_model

        self.data_loader: DataLoader = data_loader
        self.private_labels: torch.Tensor = private_labels

        self.experiment_params: ExperimentParams = experiment_params
        self.mine_params: MINE_Params = experiment_params.mine_args
        self.encoder_params = experiment_params.encoder_params
        self.beta: float = experiment_params.beta
        # Define the device
        self.device: torch.device = device



    def get_MINE(
        self,
        *,
        stats_network: nn.Module,
        transformed_data_loader: DataLoader,
        enc_out_num_nodes: int,
        train_epoch: int,
        mine_batch_size: int,
        experiment_name: str,
        device: torch.device,
        gradient_batch_size=1,
    ) -> Tuple[MutualInformationEstimator, TensorBoardLogger]:
        """
        Get the MINE model and logger

        Args:
            transformed_data_loader (DataLoader): Dataloader that contains the output of the encoder
            enc_out_num_nodes (int): Number of nodes in the encoder's output
            mine_epochs (int): Number of epochs to train the MINE model
            train_epoch (int): Current training epoch
            gradient_batch_size (int, optional): . Defaults to 1.
            func_str (str, optional): Determines how many mini batches (MINE iters) of gradients get accumulated before optimizer step gets applied. Defaults to None.
            utility (bool, optional): Determines if the MINE model is for utility or privacy. Defaults to True.

        Returns:
            Tuple[nn.Module, TensorBoardLogger]: MINE model and logger
        """

        # if utility:
        #     stats_network = self._get_utility_stats_network()
        # else:
        #     stats_network = self._get_privacy_stats_network()

        # Define Mine model
        mi_estimator = Mine(stats_network, loss="mine").to(self.device)
        func_str = (
            f"training epoch={train_epoch}: f(x)=DenseEnc(x) {enc_out_num_nodes} nodes"
        )

        kwargs = {
            "mine": mi_estimator,
            "lr": 1e-4,
            "batch_size": mine_batch_size,
            "alpha": 0.1,  # Used as the ema weight in MINE
            "func": func_str,
            "train_loader": transformed_data_loader,
            # Determines how many mini batches (MINE iters) of gradients get accumulated before optimizer step gets applied
            # Meant to stabilize the MINE curve for [hopefully] better encoder training performance
            "gradient_batch_size": gradient_batch_size,
        }

        logger = TensorBoardLogger(
            "lightning_logs",
            name=f"{experiment_name} BS={mine_batch_size}",
            version=f"{func_str}, BS={mine_batch_size}",
        )

        model = MutualInformationEstimator(loss="mine", **kwargs).to(device)
        return model, logger

    def forward(
        self,
        *,
        num_batches_final_MI,
        include_privacy=True,
        include_utility=True,
        gradient_batch_size=1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """K = MINE_BATCH_SIZE"""
        # Get encoder transformed data
        dataset: CustomDataset = self.data_loader.dataset
        transformed_embeddings: torch.Tensor = self.encoder(
            dataset.inputs.float().to(self.device)
        )

        labels_public = dataset.targets.float()
        labels_private = self.private_labels.float()

        # Define datasets for MINE

        # Public Label
        z_train_utility_detached = CustomDataset(
            transformed_embeddings.detach(),
            labels_public.detach(),  # self.data_loader.dataset.inputs.float().to(device).detach()
        )

        # Private Label
        z_train_privacy_detached = CustomDataset(
            transformed_embeddings.detach(), labels_private.detach()
        )

        # Define dataloaders for MINE
        z_train_loader_utility_detached = DataLoader(
            z_train_utility_detached, self.mine_batch_size, shuffle=True
        )
        z_train_loader_privacy_detached = DataLoader(
            z_train_privacy_detached, self.mine_batch_size, shuffle=True
        )

        # TODO: Fix this part
        # Get MINE model (sitting in Pytorch lightning module)
        model_MINE_utility: MutualInformationEstimator
        model_MINE_privacy: MutualInformationEstimator

        MINE_utility_stats_network: nn.Module = self._get_utility_stats_network()
        MINE_privacy_stats_network: nn.Module = self._get_privacy_stats_network()

        model_MINE_utility, logger_utility = self.get_MINE(
            stats_network=MINE_utility_stats_network,
            transformed_data_loader=z_train_loader_utility_detached,
            enc_out_num_nodes=self.encoder_model.out_size,
            train_epoch=self.mine_params.mine_epochs_utility,
            mine_batch_size=self.mine_params.mine_batch_size,
            experiment_name=self.experiment_params.experiment_name,
            device=self.device,
            gradient_batch_size=gradient_batch_size,
        )
        model_MINE_privacy, logger_privacy = self.get_MINE(
            stats_network=MINE_privacy_stats_network,
            transformed_data_loader=z_train_loader_privacy_detached,
            enc_out_num_nodes=self.encoder_model.out_size,
            train_epoch=self.mine_params.mine_epochs_privacy,
            mine_batch_size=self.mine_params.mine_batch_size,
            experiment_name=self.experiment_params.experiment_name,
            device=self.device,
            gradient_batch_size=gradient_batch_size,
        )

        # TODO: Fix this part
        # # If previous MINE model exists, load it
        # if self.utility_stats_network_path:
        #     try:
        #         model_MINE_utility.energy_loss.load_state_dict(
        #             torch.load(self.utility_stats_network_path)
        #         )
        #     except FileNotFoundError:
        #         print("No previous MINE model found, training from scratch")

        # if self.privacy_stats_network_path:
        #     try:
        #         model_MINE_privacy.energy_loss.load_state_dict(
        #             torch.load(self.privacy_stats_network_path)
        #         )
        #     except FileNotFoundError:
        #         print("No previous MINE model found, training from scratch")

        # Optimize MINE estimate, "train" MINE
        last_mi_utility = last_mi_privacy = 0
        if include_utility:
            trainer_utility = Trainer(
                max_epochs=self.mine_params.mine_epochs_utility,
                logger=logger_utility,
                gpus=1,
            )
            trainer_utility.fit(model_MINE_utility)

            # TODO: Fix this part
            # if self.utility_stats_network_path:
            #     print("Using MI Strategy, saving MINE model")
            #     # Save the weights of the MINE model
            #     torch.save(
            #         model_MINE_utility.energy_loss.state_dict(),
            #         self.utility_stats_network_path,
            #     )

            ## -------- Calculate I(T(x); L(x)) estimate after MINE training ---------- ##
            # **IMPORTANT**: Use the non-detached og transformed_embeddings so that gradients are retained
            z_train_utility = CustomDataset(
                transformed_embeddings, labels_public.float()
            )

            # z_train_utility = CustomDataset(
            #     transformed_embeddings, self.data_loader.dataset.inputs.float().to(device)
            # )

            z_train_loader_utility = DataLoader(
                z_train_utility, self.mine_params.mine_batch_size, shuffle=True
            )
            model_MINE_utility.energy_loss.to(self.device)
            sum_MI_utility = 0

            # Average MI across num_batches_final_MI batches to lower variance
            # Batches are K random samples from the dataset after all
            print(
                "Num batches final MI: ",
                num_batches_final_MI,
                "len dataset: ",
                len(z_train_loader_utility.dataset),
                "K: ",
                self.mine_params.mine_batch_size,
                "len dataset / K: ",
                len(z_train_loader_utility.dataset) / self.mine_batch_size,
            )
            assert num_batches_final_MI <= (
                math.ceil(len(z_train_loader_utility.dataset) / self.mine_batch_size)
            )
            utility_it = iter(z_train_loader_utility)
            for _ in range(num_batches_final_MI):
                Tx, Lx = next(utility_it)
                Tx, Lx = Tx.to(self.device), Lx.to(self.device)
                sum_MI_utility += model_MINE_utility.energy_loss(Tx, Lx)

            # MINE loss = -1 * MI estimate since we are maximizing using gradient descent still
            last_mi_utility = -1 * sum_MI_utility / num_batches_final_MI

        if include_privacy:
            trainer = Trainer(
                max_epochs=self.mine_params.mine_epochs_privacy,
                logger=logger_privacy,
                gpus=1,
            )
            trainer.fit(model_MINE_privacy)

            # # If path exists, save the weights of the MINE model
            # if self.privacy_stats_network_path:
            #     print("Using MI Strategy, saving MINE model")
            #     torch.save(
            #         model_MINE_privacy.energy_loss.state_dict(),
            #         self.privacy_stats_network_path,
            #     )

            #     ## -------- Calculate I(T(x); S(x)) estimate after MINE training ---------- ##
            z_train_privacy = CustomDataset(transformed_embeddings, labels_private)
            z_train_loader_privacy = DataLoader(
                z_train_privacy, self.mine_params.mine_batch_size, shuffle=True
            )
            model_MINE_privacy.energy_loss.to(self.device)

            assert num_batches_final_MI <= math.ceil(
                len(z_train_loader_privacy.dataset) / self.mine_params.mine_batch_size
            )
            sum_MI_privacy = 0
            privacy_it = iter(z_train_loader_privacy)
            # prev_mi = None
            for i in range(num_batches_final_MI):
                Tx: torch.Tensor
                Sx: torch.Tensor
                Tx, Sx = next(privacy_it)
                Tx, Sx = Tx.to(self.device), Sx.to(self.device)
                sum_MI_privacy += model_MINE_privacy.energy_loss(Tx, Sx)

            last_mi_privacy: torch.Tensor = -1 * sum_MI_privacy / num_batches_final_MI

        print(
            f"final MI values: utility: {last_mi_utility}, privacy: {last_mi_privacy}"
        )
        return last_mi_utility, last_mi_privacy

    def train_encoder(
        self,
        *,
        num_batches_final_MI: int = 100,
        include_privacy: bool = True,
        include_utility: bool = True,
        gradient_batch_size: int = 1,
    ):
        """K = MINE_BATCH_SIZE"""
        # Encoder's training params
        mi_utility: torch.Tensor
        mi_privacy: torch.Tensor
        learning_rate = 1e-3

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder_model.parameters(),
            lr=learning_rate,
        )
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
            if self.encoder_params.enc_save_path is not None:
                self._save_encoder_weights(epoch)

            print(f"====> Epoch: {epoch} Utility MI I(T(x); L(x)): {mi_utility:.8f}")
            print(f"====> Epoch: {epoch} Privacy MI I(T(x); S(x)): {mi_privacy:.8f}")
            print(f"====> Epoch: {epoch} Loss: {loss:.8f}")

    def _get_utility_stats_network(self) -> nn.Module:
        stats_network = vars(models_to_train)[
            self.mine_params.utility_stats_network_model_name
        ](**self.mine_params.utility_stats_network_model_params)
        return stats_network

    def _get_privacy_stats_network(self) -> nn.Module:
        stats_network = vars(models_to_train)[
            self.mine_params.privacy_stats_network__model_name
        ](**self.mine_params.privacy_stats_network_model_params)
        return stats_network

    def _save_encoder_weights(self, epoch: int) -> None:
        enc_save_dir_path: Path = self.encoder_params.enc_save_dir_path

        # Don't save the state dict since that doesn't include the model parameters + their gradients
        # Options were to save entire model or optimizer's state dict:
        # https://discuss.pytorch.org/t/how-to-save-the-requires-grad-state-of-the-weights/52906/6
        print(f"Saving weights to {enc_save_dir_path}")

        # Check if the enc_save_path exists
        if not enc_save_dir_path.exists():
            enc_save_dir_path.mkdir(parents=True, exist_ok=True)

        # TODO: Fix this part
        encoder_model_path = enc_save_dir_path / f"{self.experiment_params.experiment_name}-epoch={epoch}.pt"
        torch.save(
            self.encoder_model,
            encoder_model_path,
        )
        
        optimizer_path = enc_save_dir_path / f"[optimizer] {self.experiment_params.experiment_name}-epoch={epoch}.pt"
        torch.save(
            self.encoder_optimizer.state_dict(),
            optimizer_path,
        )

    # def save_mi_scores(self, dir):
    #     # Check if the directory exists
    #     if not os.path.exists(dir):
    #         os.makedirs(dir, exist_ok=True)

    #     utility_scores = np.array(self.utility_scores)
    #     privacy_scores = np.array(self.privacy_scores)

    #     # Save
    #     utility_save_path = os.path.join(dir, f"{self.experiment_name}-utility.npy")
    #     privacy_save_path = os.path.join(dir, f"{self.experiment_name}-privacy.npy")

    #     np.save(utility_save_path, utility_scores)
    #     np.save(privacy_save_path, privacy_scores)

if __name__ == "__main__":
    print("Test")