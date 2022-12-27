"""
Parts for the model.

@version 2022-12
@author Raúl Coterillo
"""

# lightning
from pytorch_lightning import LightningModule

import torch.nn as nn
import torch

# ========================================================= #
#                     NN BASIC MODULES                      #
# ========================================================= #

class LinSeq(LightningModule):

    """ Basic linear sequence. """

    def __init__(self, 
            in_features: int,
            hid_features: int,
            out_features: int,
            hid_layers: int = 0
        ) -> None:

        super().__init__()
        self.save_hyperparameters()
        
        self.in_features = in_features
        
        self.hid_features = hid_features
        self.out_features = out_features

        layers = []
        layers.append(nn.Linear(in_features=in_features, out_features=hid_features))
        for _ in range(hid_layers):
            layers.append(nn.Linear(in_features=hid_features, out_features=hid_features)) 
        layers.append(nn.Linear(in_features=hid_features, out_features=out_features))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

class ConvEncoder(LightningModule): 

    """ Convolutional encoder, shared by all tasks. """

    def __init__(self, 
        img_channels = 5,
        layers = [64,64,64,64,64,64], 
        cstride: int = 1,
        ckern_size: int = 3,
        pkern_size: int = 2, 
        dropout: float = 0.0
        ) -> None:

        super().__init__()
        self.save_hyperparameters()
        
        self.cstride = cstride
        self.ckern_size = ckern_size

        self.pkern_size = pkern_size
        
        self.dropout = dropout

        def conv_block(input_size, output_size):
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size=ckern_size, 
                    stride=cstride, padding="same"), 
                nn.BatchNorm2d(output_size),
                nn.MaxPool2d(kernel_size=pkern_size),
                nn.ReLU(), nn.Dropout(dropout))

        self.test_block = conv_block(5,16)

        num_features = img_channels
        self.features = nn.Sequential()
        for i, filters in enumerate(layers):
            self.features.add_module(name=f"clayer_{i}", module=conv_block(input_size=num_features, output_size=filters))
            num_features = filters

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        return self.flatten(self.features(x)) 

    def get_encoder_features(self):
        image_dim = (1, self.in_channels, self.img_height, self.img_width)
        features = self.encoder(torch.rand(image_dim).float())
        return features.view(features.size(0), -1).size(1)

    def get_output_shape(self):
        return self.out_channels*2


