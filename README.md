# Datathon AI4ES Reto A

satélites, cosechadores, tres o cuatro (empresas) explotadoras 🎶 

## Description (by ChatGPT)

Artificial intelligence (AI) has made significant strides in recent years, particularly in the field of image recognition. One of the most promising applications of AI in image recognition is the detection of crop diseases from aerial drone images.

Crop diseases can have a devastating impact on agriculture, leading to significant losses in yield and revenue. Early detection of crop diseases is crucial for mitigating their effects and implementing control measures. However, manually detecting crop diseases is a time-consuming and labor-intensive process that requires trained professionals to visually inspect crops for signs of disease.

AI-based image recognition offers a solution to this problem by automating the detection process. Using aerial drone images, AI algorithms can analyze large areas of crops and identify the presence of diseases with a high degree of accuracy. This not only saves time and resources, but it also allows for more frequent monitoring, which can lead to early detection and effective control measures.

In addition to its practical benefits, AI-based image recognition also has the potential to improve the accuracy and consistency of crop disease detection. Human inspection is subject to human error and can be influenced by factors such as fatigue and personal bias. AI algorithms, on the other hand, are consistent and unbiased, making them a reliable and accurate tool for crop disease detection.

In conclusion, AI-based image recognition has the potential to revolutionize the way we detect and control crop diseases. By automating the process, we can more efficiently and effectively identify and address these threats to agriculture, ultimately leading to improved crop yields and sustainability.

## Environment / Setup

```bash
git clone https://github.com/rcote98/ai4es-datathon-retoA   # clone the repo
cd ai4es-datathon-retoA                                     # move in the folder
python3 -m venv ai4es_env                                   # create virtualenv
source ai4es_env/bin/activate                               # activate it
pip install -r requirements.txt                             # install dependencies
```

## Estructure

- **data.py:** Carga los datos de las imágenes.
- **network.py:** Estructura de la red neuronal.
- **train.py:** Script de entrenamiento de la red.

## Visualize Training Progress
```bash
tensorboard --logdir=[whatever-dir-name]/lightning_logs/
```
