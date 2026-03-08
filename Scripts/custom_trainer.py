"""
Custom nnU-Net Trainer with tqdm progress bars and frequent checkpointing.

Place this file in the nnunetv2 trainer variants directory so it gets
auto-discovered. Use with:  -tr nnUNetTrainerProgress
"""

import torch
import numpy as np
from os.path import join
from time import time
from tqdm import tqdm

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerProgress(nnUNetTrainer):
    """nnUNetTrainer with:
    - tqdm progress bars (epoch + batch level)
    - More frequent checkpointing (every 25 epochs)
    - Numbered epoch checkpoints every 100 epochs
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Reduce total epochs (default is 1000; 250 is enough for 50 training cases)
        self.num_epochs = 250
        # Checkpoint every 25 epochs instead of 50
        self.save_every = 25
        # Save a permanent numbered checkpoint every N epochs
        self.save_numbered_every = 50

    def run_training(self):
        """Override to add tqdm progress bars around the epoch and batch loops."""
        self.on_train_start()

        # Epoch-level progress bar
        epoch_pbar = tqdm(
            range(self.current_epoch, self.num_epochs),
            desc="Training",
            unit="epoch",
            initial=self.current_epoch,
            total=self.num_epochs,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for epoch in epoch_pbar:
            self.on_epoch_start()

            # ── Training phase ──
            self.on_train_epoch_start()
            train_outputs = []
            train_pbar = tqdm(
                range(self.num_iterations_per_epoch),
                desc=f"  Epoch {epoch} [train]",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for batch_id in train_pbar:
                output = self.train_step(next(self.dataloader_train))
                train_outputs.append(output)
                # Show running loss
                if "loss" in output:
                    train_pbar.set_postfix(loss=f"{output['loss']:.4f}")
            train_pbar.close()
            self.on_train_epoch_end(train_outputs)

            # ── Validation phase ──
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                val_pbar = tqdm(
                    range(self.num_val_iterations_per_epoch),
                    desc=f"  Epoch {epoch} [val]  ",
                    unit="batch",
                    leave=False,
                    dynamic_ncols=True,
                )
                for batch_id in val_pbar:
                    output = self.validation_step(next(self.dataloader_val))
                    val_outputs.append(output)
                val_pbar.close()
                self.on_validation_epoch_end(val_outputs)

            # ── Epoch end (checkpointing, logging) ──
            self.on_epoch_end()

            # Update epoch bar with metrics
            try:
                train_loss = self.logger.my_fantastic_logging['train_losses'][-1]
                val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
                dice = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
                epoch_pbar.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    dice=f"{dice:.4f}",
                )
            except (IndexError, KeyError):
                pass

        epoch_pbar.close()
        self.on_train_end()

    def on_epoch_end(self):
        """Override to add numbered checkpoint saves at fixed intervals."""
        # Call the parent on_epoch_end (handles periodic + best checkpointing)
        super().on_epoch_end()

        # Save numbered checkpoints for milestone epochs (note: current_epoch
        # was already incremented by super().on_epoch_end())
        epoch_just_finished = self.current_epoch - 1
        if (epoch_just_finished + 1) % self.save_numbered_every == 0:
            fname = join(self.output_folder, f"checkpoint_epoch_{epoch_just_finished + 1:04d}.pth")
            self.print_to_log_file(f"Saving numbered checkpoint: {fname}")
            self.save_checkpoint(fname)
