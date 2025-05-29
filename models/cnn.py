import sys

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, MaxPool2d, Conv2d, Flatten, Dropout, BatchNorm2d


class BaseBlock(nn.Module):
    """A base CNN block.

    Can be constructed as an arbitrary composition of convolutional and pooling layers.
    """

    def __init__(self, in_channels: int, out_channels: int, conv_kernel: int = 3, conv_padding: int = 1,
                 conv_stride: int = 1, max_pool_kernel: int = 2, max_pool_stride: int = 2, conv_num=2):
        super(BaseBlock, self).__init__()
        layers = []

        for i in range(conv_num):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(
                Conv2d(conv_in, out_channels, kernel_size=conv_kernel, padding=conv_padding, stride=conv_stride)
            )
            # layers.append(BatchNorm2d(out_channels))
            layers.append(ReLU())

        layers.append(MaxPool2d(kernel_size=max_pool_kernel, stride=max_pool_stride))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Base block forward propagation """
        return self.layers(inputs)


class CNN(nn.Module):
    """A custom CNN model."""

    def __init__(self, config, in_channels: int, classes_num: int):
        super(CNN, self).__init__()
        self.config = config

        self._init_layers(in_channels, classes_num)

        self.init_function = {
            'linear': getattr(nn.init, self.config.params.linear.init_type.name + '_'),
            'convolutional': getattr(nn.init, self.config.params.convolutional.init_type.name + '_')
        }

        self.apply(self._init_weights)

    def _init_layers(self, in_channels: int, classes_num: int):
        """CNN layers initialization."""
        # Convolutions (features)
        layers = self.config.cnn.layers['features']
        layers[0]['specs']['in_channels'] = in_channels
        features = [getattr(sys.modules[__name__], layer['type'].name)(**layer['specs']) for layer in layers]

        # Classifier
        layers = self.config.cnn.layers['classifier']
        layers[-1]['specs']['out_features'] = classes_num
        classifier = [getattr(sys.modules[__name__], layer['type'].name)(**layer['specs']) for layer in layers]

        self.layers = nn.Sequential(*features, *classifier)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        """Layer parameters initialization.

        Args:
            module: A model layer.
        """
        if isinstance(module, (Linear, Conv2d)):
            module_type = {'Linear': 'linear', 'Conv2d': 'convolutional'}[module._get_name()]
            self.init_function[module_type](module.weight, **self.config.params[module_type].init_kwargs)

            if module.bias is not None:
                if self.config.params[module_type].zero_bias:
                    nn.init.zeros_(module.bias)
                else:
                    self.init_function[module_type](module.bias, **self.config.params[module_type].init_kwargs)
        elif isinstance(module, BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation implementation.

        This method propagates inputs through all the model layers.

        Args:
            inputs: A torch.Tensor with shape (batch_size, channels_num, height, width).

        Returns:
            A torch.Tensor with shape (batch_size, classes_num).
        """
        return self.layers(inputs)
