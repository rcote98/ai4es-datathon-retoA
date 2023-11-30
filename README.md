# AI4ES Datathon 2023 - Reto A - [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains my team's (Aprende Máquina) code submission for the AI4ES Datathon 2022, pertaining to challenge A. The challenge consisted on predicting percentage of diseased crops in aerial images. The images consisted of 3 channels (RGB) and were varying in size, covering only plot sections (no paths, words, etc.). The labels consisted on the percentage of diseased crops in the plot section, with no pixel-level information. The dataset was provided by the organization, and is no longer available due to privacy reasons.

## Approach

We tackled the problem as a regression problem, using a CNN to predict the percentage of diseased crops in the plot section. We used a [DenseNet](https://arxiv.org/abs/1608.06993) architecture with a custom head, MSE loss and Adam optimizer. We used [PyTorch](https://pytorch.org/) to implement the model, and [PyTorch Lightning](https://www.pytorchlightning.ai/) to train it, winning the 1st place in the Future Talent category.

## Replicating the results

To replicate the results, one would only need to install the environment specified in `requirements.txt` and run the `train.py` and `inference.py` scripts.

```sh
pip install -r requirements.txt 
```