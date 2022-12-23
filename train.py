"""
Trains the neural network.

Quite self explanatory.

@author Raúl Coterillo
@version ??-12-2022
"""


from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import LightningModule

from network_v0 import PredictionNetwork_V0
from network_v1 import PredictionNetwork_V1

from data import ImageDataModule
from pathlib import Path
import time

# Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

FOLDER = Path("models/test2")

RANDOM_STATE = 0
seed_everything(RANDOM_STATE)

SETUPS = {
    "v0": [PredictionNetwork_V0, "safe/v0_best.ckpt", False],
    "v1": [PredictionNetwork_V1, "safe/v1_best.ckpt", False]
}

LOAD_BEST = True
SETUP = SETUPS["v1"]

# Data Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Computing dataset...")
start_time = time.perf_counter()
dm = ImageDataModule(
    csv_file="csvs/train.csv",
    test_size=0.2,
    eval_size=0.2,
    batch_size=64,
    dataset_sample=1,
    random_state=RANDOM_STATE,
    image_shape=[110,330]
)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

# Model Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Creating model...", end="")
start_time = time.perf_counter()
model: LightningModule = SETUP[0](
    learning_rate=1e-5
)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

if LOAD_BEST:
    model = model.load_from_checkpoint(SETUP[1])

# Trainer Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Setup the trainer...")
start_time = time.perf_counter()

early_stop = EarlyStopping(monitor="val_mae", mode="min", patience=5)
lr_monitor = LearningRateMonitor(logging_interval='step')
model_checkpoint = ModelCheckpoint(FOLDER, save_last=True)

trainer = Trainer(
    accelerator="gpu",
    default_root_dir=FOLDER,
    callbacks=[lr_monitor, model_checkpoint, early_stop],
    log_every_n_steps=1,
    check_val_every_n_epoch=1,
    max_epochs=200, 
    deterministic = True
)

end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

# Training time!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Begin training...")
trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm)
trainer.test(model, datamodule=dm)