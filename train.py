"""
Trains the neural network.

@author Raúl Coterillo
@version 01-2023
"""

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from network import PredictionModel

from data import ImageDataModule
from pathlib import Path
import time

# Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

DESTINATION_FOLDER = Path("training")

USE_FLAGS = False
LOAD_BEST = False

RANDOM_STATE = 0
seed_everything(RANDOM_STATE)

# Data Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Computing dataset...")
start_time = time.perf_counter()
dm = ImageDataModule(
    test_size=0.1,
    eval_size=0.1,
    batch_size=48,
    dataset_sample=1,
    random_state=RANDOM_STATE,
    image_shape=[110,330],
    use_flags=USE_FLAGS,
)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

# Model Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Creating model...", end="")
start_time = time.perf_counter()
model = PredictionModel(
    num_classes=3,
    flags_size=dm.flags_size,
    learning_rate=1e-5)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

if LOAD_BEST:
    model_file = "safe/extra.ckpt" if USE_FLAGS else "safe/normal.ckpt"
    model = model.load_from_checkpoint(model_file)

# Trainer Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Setup the trainer...")
start_time = time.perf_counter()

early_stop = EarlyStopping(monitor="val_mae", mode="min", patience=5)
lr_monitor = LearningRateMonitor(logging_interval='step')
model_checkpoint = ModelCheckpoint(DESTINATION_FOLDER , monitor="val_mae", mode="min", every_n_epochs=1)

trainer = Trainer(
    accelerator="auto",
    default_root_dir=DESTINATION_FOLDER,
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