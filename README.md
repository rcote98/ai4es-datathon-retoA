# Datathon AI4ES Reto A
satélites, cosechadores, tres o cuatro (empresas) explotadoras 🎶 

## Environment / Setup

```bash
git clone https://github.com/rcote98/ai4es-datathon-retoA   # clone the repo
cd ai4es-datathon-retoA                                     # move in the folder
python3 -m venv ai4es_env                                   # create virtualenv
source ai4es_env/bin/activate                               # activate it
pip install -r requirements.txt                             # install dependencies
#python -m pip install -e .                                 # install dev package (espero que no)
```

## Visualize Training Progress
```bash
tensorboard --logdir=[whatever-dir-name]/lightning_logs/
```