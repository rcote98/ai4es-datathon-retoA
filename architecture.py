"""
Architecture of the Neural Network.

Modified version of the DenseNet PyTorch implementation presented in
https://amaarora.github.io/2020/08/02/densenets.html (original paper:
https://arxiv.org/abs/1608.06993).

Modifications:
 - increased input channels from 3 to 5
 - decreased growth rate from 64 to 12 (similar performance, less parameters)

@version 2022-12
@author Raúl Coterillo
"""

from __future__ import annotations
import torch.nn.functional as F
import torch.nn as nn
import torch

class _Transition(nn.Sequential):
    """ Transition Layer. """
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseLayer(nn.Module):
    """ Dense Layer. """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        """ Bottleneck function. """
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input): 
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    """ Dense Block. """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module):
    
    def __init__(self, 
            num_classes: int = 3,
            growth_rate: int = 12, 
            block_config: list[int] = (6, 12, 24, 16),
            num_init_features: int =64, 
            bn_size: int = 4, 
            drop_rate: float = 0,
            flags_size: int = None
            ) -> None:

        super(DenseNet, self).__init__()

        # Convolution and pooling part from table-1
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(5, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Add multiple denseblocks based on config 
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layers
        self.num_classes = num_classes
        self.features_size = num_features
        self.flags_size = flags_size if flags_size is not None else 0
        fts, fgs = self.features_size, self.flags_size

        self.lin1 = nn.Linear(in_features=fts+fgs, out_features=fts//2)
        
        self.lin2 = nn.Sequential()
        self.lin2.add_module("lin0", nn.Linear(in_features=fts//2 + fgs, out_features=num_classes))    
        self.lin2.add_module("sig0", nn.Sigmoid(inplace=True))

        # Official weight initizalization from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # if no flags are used
        if self.flags_size is None:
            
            out = self.features(x)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            
            out = self.lin2(self.lin1(out))
        
        # if flags are used
        else:
            image, flags = x[0], x[1]

            out = self.features(image)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)

            out = torch.cat((out, flags), 1)
            out = self.lin1(out)
            out = torch.cat((out, flags), 1)
            out = self.lin2(out)

        return out
