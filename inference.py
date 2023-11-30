"""
Generates prediction for the test dataset in the competition format.

@author Raúl Coterillo
@version 12-2022
"""

from pytorch_lightning import Trainer
from network import PredictionModel
from data import ImageDataModule

import pandas as pd
import numpy as np

if __name__=="__main__":

    # inicializa el dataset
    dm = ImageDataModule(pred_csv_file="csvs/test.csv", use_flags=False)

    # carga el modelo y crea el trainer
    model, trainer = PredictionModel.load_from_checkpoint("safe/normal_mod.ckpt"), Trainer(accelerator="auto", logger=[])
    
    # genera las predicciones
    predictions = trainer.predict(model, datamodule=dm)                     # devuelve una lista de tensores (resultados)
    predictions = np.array([x.numpy().squeeze() for x in predictions])*100  # convierte a array y escala a porcentajes
    predictions = np.round(predictions, decimals=0)                         # redondea a los decimales de los datos de entrada

    # guarda los resultados
    version = "v2"
    participant = "Aprende_Máquina"
    test_df = pd.read_csv("csvs/test.csv")
    test_df[["DISEASE1", "DISEASE2", "DISEASE3"]] = predictions
    results_filename = '-'.join([participant, 'test_results', version])
    test_df.to_csv(results_filename, index=False)

