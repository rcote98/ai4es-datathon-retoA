"""


@author Raúl Coterillo
@version 12-2022
"""

from pytorch_lightning import Trainer
from network import PredictionModel
from data import ImageDataModule

import pandas as pd
import numpy as np

def predict(image):
    import random
    return [random.randrange(0, 101), random.randrange(0, 101), random.randrange(0, 101)]

if __name__=="__main__":

    USE_FLAGS = True

    test_df = pd.read_csv("test.csv")

    dm = ImageDataModule(pred_csv_file="csvs/test.csv", use_flags=True)
    model_file = "save/best.ckpt" if USE_FLAGS else "save/best_no_flags.ckpt"

    model, trainer = PredictionModel.load_from_checkpoint(model_file), Trainer()
    predictions = trainer.predict(model, datamodule=dm)

    participant = "Aprende_Máquina"
    version = "v1"
    results_filename = '-'.join([participant, 'test_results', version])
    test_df.to_csv(results_filename, index=False)

