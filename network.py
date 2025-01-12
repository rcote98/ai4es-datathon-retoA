"""
Neural network.

Does neural network stuff.

@author Raúl Coterillo
@version ??-12-2022
"""


# lightning imports
from pytorch_lightning import LightningModule

# torch imports
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics as tm
import torch.nn as nn
import torch

import numpy as np

from architecture import DenseNet

# =========================================================================== #
# =========================================================================== #

class PredictionModel(LightningModule):

    def __init__(self,     
        num_classes: int = 3,
        flags_size: int = None,
        learning_rate: float = 1e-5,
        ):

        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.flags_size = flags_size

        self.densenet = DenseNet(flags_size=self.flags_size)
        
        for phase in ["train", "val", "test"]:
                self.__setattr__(f"{phase}_r2", tm.R2Score(num_outputs=3))
                self.__setattr__(f"{phase}_mae",  tm.MeanAbsoluteError())

    def forward(self, x: torch.Tensor):
        """ Use for inference only (separate from training_step)"""
        return self.densenet(x)

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _inner_step(self, batch: tuple[torch.Tensor], stage: str = None):

        """ Common actions for training, test and eval steps. """
        
        x, y = batch
        results = self(x)    
        loss = nn.functional.mse_loss(results, y)

        r2 = self.__getattr__(f"{stage}_r2")(results, y)
        mae = self.__getattr__(f"{stage}_mae")(results, y)
            
        if stage == "train":
            self.log(f"{stage}_loss", loss, sync_dist=True)
            self.log(f"{stage}_r2", r2, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_mae", mae, prog_bar=True, sync_dist=True)

        return loss.to(torch.float32)

    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        """ Prediction step. """
        return self(batch)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        """ Test step. """
        return self._inner_step(batch, stage="test")

    # EPOCH END
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _custom_epoch_end(self, step_outputs, stage):

        """ Common actions for validation and test epoch ends. """

        if stage != "train":
            loss = torch.tensor(step_outputs).mean()
            self.log(f"{stage}_loss", loss, sync_dist=True)

        # metrics to analyze
        metrics = ["r2", "mae"]

        print(f"\n\n  ~~ {stage} stats ~~")
        for metric in metrics:
            mstring = f"{stage}_{metric}"

            val = self.__getattr__(mstring).compute()
            if stage == "train":
                self.log("epoch_" + mstring, val, sync_dist=True)
            else:
                self.log(mstring, val, sync_dist=True)

            self.__getattr__(mstring).reset()
            print(f"{mstring}: {val:.4f}")
        print("")

    def training_epoch_end(self, training_step_outputs):
        """ Actions to carry out at the end of each training epoch. """
        self._custom_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        """ Actions to carry out at the end of each validation epoch. """
        self._custom_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        """ Actions to carry out at the end of each test epoch. """
        self._custom_epoch_end(test_step_outputs, "test")

    # OPTIMIZERS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def configure_optimizers(self):

        """ Define optimizers and LR schedulers. """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


# =========================================================================== #
# =========================================================================== #
