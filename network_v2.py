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

from densenet import DenseNet

# =========================================================================== #
# =========================================================================== #

class PredictionNetwork_V2(LightningModule):

    def __init__(self,     
        cameras: int = 2,
        num_classes: int = 3,
        learning_rate: float = 1e-5,
        ):

        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # # main encoder
        # self.encoder = ConvEncoder(
        #     in_channels=5,
        #     out_channels=128,
        #     conv_kernel_size=3,
        #     pool_kernel_size=3,
        #     img_height=110,
        #     img_width=330
        # )


        # encoder_out_feats = self.encoder.get_output_shape()

        # self.decoder = nn.Sequential(
        #     LinSeq(in_features=encoder_out_feats,
        #     hid_features=encoder_out_feats*2,
        #     out_features=3,
        #     hid_layers=0), nn.Sigmoid())

        self.densenet = DenseNet(growth_rate=12, num_classes=3)
        buffer =  cameras + 12

        feats = self.densenet.num_features + buffer
        self.lin1 =  nn.Linear(in_features=feats, out_features=feats//2, dtype=torch.float32)
        self.lin2 = nn.Linear(in_features=feats//2 + buffer, out_features=num_classes, dtype=torch.float32)
        
        for phase in ["train", "val", "test"]:
                # self.__setattr__(f"{phase}_r2", tm.R2Score())
                self.__setattr__(f"{phase}_mae",  tm.MeanAbsoluteError())

    def forward(self, x: torch.Tensor):
        """ Use for inference only (separate from training_step)"""
        #return self.decoder(self.encoder(x))
        image, month, camera = x[0].type(torch.float32), x[1].type(torch.float32), x[2].type(torch.float32)        
        out = self.densenet(image)
        out = torch.cat((out, month, camera))
        out = self.lin1(out)
        out = torch.cat((out, month, camera))
        out = self.lin2(out)
        return out

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # def predict_step(self, batch, batch_idx):

    #     """ Prediction step, skips auxiliary tasks. """

    #     if self.tasks.main:
    #         shared = self.conv_encoder(batch)
    #         result = self.main_decoder(shared)
    #         return result
    #     else:
    #         raise NotImplementedError()

    def _inner_step(self, batch, stage: str = None):

        """ Common actions for training, test and eval steps. """

        # x[0] is the time series
        # x[1] are the sim frames
        
        x, y = batch

        results = self(x)    
        loss = nn.functional.mse_loss(results, y)

        #r2 = self.__getattr__(f"{stage}_r2")(results, y)
        mae = self.__getattr__(f"{stage}_mae")(results, y)
            
        if stage == "train":
            self.log(f"{stage}_loss", loss, sync_dist=True)
            # self.log(f"{stage}_r2", r2, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_mae", mae, prog_bar=True, sync_dist=True)

        return loss.to(torch.float32)

    def training_step(self, batch, batch_idx):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch, batch_idx):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        """ Test step. """
        return self._inner_step(batch, stage="val")

    # EPOCH END
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _custom_epoch_end(self, step_outputs, stage):

        """ Common actions for validation and test epoch ends. """

        if stage != "train":
            loss = torch.tensor(step_outputs).mean()
            self.log(f"{stage}_loss", loss, sync_dist=True)

        # metrics to analyze
        metrics = ["mae"]#, "r2"]

        for metric in metrics:
            mstring = f"{stage}_{metric}"

            val = self.__getattr__(mstring).compute()
            if stage == "train":
                self.log("epoch_" + mstring, val, sync_dist=True)
            else:
                self.log(mstring, val, sync_dist=True)

            self.__getattr__(mstring).reset()
            print(f"\n{mstring}: {val:.4f}")
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
