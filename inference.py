


from data import ImageDataset
from network import PredictionModel

import numpy as np
import pandas as pd

from pytorch_lightning import Trainer
import torch


def predict(image):
    import random
    return [random.randrange(0, 101), random.randrange(0, 101), random.randrange(0, 101)]


if __name__=="__main__":

    USE_FLAGS = True

    test_df = pd.read_csv("test.csv")
    test_ds = ImageDataset(dataset=test_df, image_shape=[110,330], use_flags=USE_FLAGS)

    model_file = "save/best.ckpt" if USE_FLAGS else "save/best_no_flags.ckpt"

    model, trainer = PredictionModel.load_from_checkpoint(model_file), Trainer()
    predictions = trainer.predict(model, dataloaders=test_ds)

    for image_path in test_df["PLOT_FILE"]:
        image = load_image(image_path=image_path, remove_negs=True, normalization=True)
        prediction = predict(image=image)
        test_df.loc[test_df.PLOT_FILE == image_path, ['DISEASE1', 'DISEASE2', 'DISEASE3']] = prediction
    
    
    participant = "Aprende_Máquina"
    version = "v1"
    results_filename = '-'.join([participant, 'test_results', version])
    test_df.to_csv(results_filename, index=False)

