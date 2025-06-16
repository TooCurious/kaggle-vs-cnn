import sys
from typing import Type, Union, List

import torch
import torch.nn.functional as F
from torch import nn


class BaseBlock(nn.Module):
    """A base block used as a building block for ResNet-18/34 models.

    This block is usually repeated several times within each of the model stages.

    At each repeat the block has two paths for the input:
        - Path A: Conv -> BN -> ReLU -> Conv -> BN
                  Two convolutions defined as follows:
                      1. The same depth for both convolutions (depth is specified in model_cfg.resnet.layers for each stage)
                      2. The same kernel size = (3, 3) for both convolutions
                      3. The same padding = 1 for both convolutions
                      4. For the second convolution: stride = 1 always
                      5. For the first convolution: stride = 2 only at first repeat at all stages except for the first one,
                                and stride = 1 otherwise

        - Path B: Identity or (Conv -> BN)
                  A shortcut connection is defined as follows:
                      1. 'Downsampling' layer that consists of one convolution with the same depth as those from path A,
                                kernel = (1, 1) and stride = 2 only at first repeat at all stages except for the first one
                      2. Identity layer (nn.Identity) at all other repeats

    Inputs are passed through the both paths, then results should be summed up and passed through ReLU function.
    """

    def __init__(self, in_channels: int, block_out_channels: int = 64, stride: int = 1, expansion: int = 1):
        """
        Args:
            in_channels: The number of input channels.
            block_out_channels: The number of output channels for the first block,
                    number of output channels for the whole building block is block_channels * expansion.
            stride: Stride for the first convolution in path A and convolution in path B (in 'Downsampling' case).
            expansion: The factor by which block_channels is multiplied to get the whole block output channels number.
        """
        super(BaseBlock, self).__init__()
        if in_channels == block_out_channels * expansion:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, block_out_channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block_out_channels * expansion)
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, block_out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(block_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                block_out_channels, block_out_channels * expansion, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(block_out_channels * expansion)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        identity = inputs
        out = self.conv_block(inputs)
        out += self.shortcut(identity)
        return F.relu(out)


class BottleneckBlock(nn.Module):
    """A 'bottleneck' block used as a building block for ResNet-50/101/152

     This block is usually repeated several times within each of the model stages.

    At each repeat the block has two paths for the input:
        - Path A: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> BN
                  Three convolutions defined as follows:
                      1. The same depth for the first two convolutions and depth x 4 (model_cfg.resnet.expansion_factor)
                            for the third convolution (depth is specified in model_cfg.resnet.layers for each stage)
                      2. (1, 1) kernel size for the first and the third convolutions and (3, 3) - for the second
                      3. Padding = 1 only for the second convolution and 0 for all others
                      4. For the first and the third convolutions: stride = 1 always
                      5. For the second convolution: stride = 2 only at first repeat at all stages except for the first one,
                            and stride = 1 otherwise

        - Path B: Identity or (Conv -> BN)
                  A shortcut connection is defined as follows:
                      1. 'Downsampling' layer that consists of one convolution with the same depth as for the last
                            convolution from path A, kernel = (1, 1) and stride = 2 only at first repeat at all stages
                            except for the first one
                      2. Identity layer (nn.Identity) at all other repeats

    Inputs are passed through the both paths, then results should be summed up and passed through ReLU function.
    """

    def __init__(self, in_channels: int, block_out_channels: int = 64, stride: int = 1, expansion: int = 4):
        """
        Args:
            in_channels: The number of input channels.
            block_out_channels: The number of output channels for the first two blocks, number of output channels
                     for the third convolution and the whole building block is block_channels * expansion.
            stride: Stride for the second convolution in path A and convolution in path B (in 'Downsampling' case).
            expansion: The factor by which block_channels is multiplied to get the whole building block output channels number.
        """
        super(BottleneckBlock, self).__init__()
        if in_channels == block_out_channels * expansion:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, block_out_channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block_out_channels * expansion)
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, block_out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_out_channels),
            nn.ReLU(),
            nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(block_out_channels),
            nn.ReLU(),
            nn.Conv2d(block_out_channels, block_out_channels * expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_out_channels * expansion)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        identity = inputs
        out = self.conv_block(inputs)
        out += self.shortcut(identity)
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet model.

    A class for implementing ResNet-18/34/50/101/152 from
        'Deep Residual Learning for Image Recognition' (https://arxiv.org/abs/1512.03385)
    """

    def __init__(self, config, in_channels: int, classes_num: int):
        super(ResNet, self).__init__()
        self.config = config

        self._init_layers(in_channels, classes_num)

        self.init_function = {
            'linear': getattr(nn.init, self.config.params.linear.init_type.name + '_'),
            'convolutional': getattr(nn.init, self.config.params.convolutional.init_type.name + '_')
        }
        self.apply(self._init_weights)

    def _make_stem(self, in_channels: int) -> (list, int):
        """Makes stem module.

        Scheme: Conv -> BN -> ReLU -> MaxPool

        A stem module is a block with one convolution with <in_channels> input channels number,
            <output_channels> output channels number, kernel size (7, 7), stride = 2 and padding = 3,
            followed by a Batch normalization layer, ReLU function and Max Pool layer with kernel size = (3, 3),
            stride = 2 and padding = 1.

        Args:
            in_channels: The number of input channels.
        """
        if self.config.resnet.stem is not None:
            out_channels = self.config.resnet.stem
            stem = [
                nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
        else:
            stem, out_channels = [], in_channels
        return stem, out_channels

    def _make_stages(self, previous_output: int) -> (list, int):
        """Makes stages from residual blocks.

        General scheme: stage 1 -> stage 2 -> stage 3 -> stage 4

        This method gather all stages with repeated building blocks.

        Args:
            previous_output: The number of input channels for the current stage.
        """
        stages, expansion = [], self.config.resnet.expansion_factor

        for stage in self.config.resnet.layers['stages']:
            stage_specs = stage['specs'].copy()
            building_block = getattr(sys.modules[__name__], stage['type'].name)
            stages.append(building_block(previous_output, expansion=expansion, **stage_specs))
            previous_output = stage_specs['block_out_channels'] * expansion
            stage_specs['stride'] = 1

            for _ in range(stage['block_repeats'] - 1):
                stages.append(building_block(previous_output, expansion=expansion, **stage_specs))

        return stages, previous_output

    @staticmethod
    def _make_classifier(previous_output: int, classes_num: int) -> list:
        """Makes classification layer.

        Scheme: AvgPool -> Flatten -> Linear

        Args:
            previous_output: The number of input channels.
            classes_num: The number of classes.
        """
        classifier = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=previous_output, out_features=classes_num)
        ]
        return classifier

    def _init_layers(self, in_channels: int, classes_num: int):
        """Layers initialization.

        Args:
            in_channels: The number of model input channels.
            classes_num: The number of classes.
        """
        # Stem (layers before the residual blocks)
        stem, output_channels = self._make_stem(in_channels)

        # Stages (residual blocks)
        stages, output_channels = self._make_stages(output_channels)

        # Classifier
        classifier = self._make_classifier(output_channels, classes_num)

        self.layers = nn.Sequential(*stem, *stages, *classifier)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        """Layer parameters initialization.

        Args:
            module: A model layer.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module_type = {'Linear': 'linear', 'Conv2d': 'convolutional'}[module._get_name()]
            self.init_function[module_type](module.weight, **self.config.params[module_type].init_kwargs)

            if module.bias is not None:
                if self.config.params[module_type].zero_bias:
                    nn.init.zeros_(module.bias)
                else:
                    self.init_function[module_type](module.bias, **self.config.params[module_type].init_kwargs)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation implementation.

        This method propagates inputs through all the model layers.

        Args:
            inputs: A torch.Tensor with shape (batch_size, channels_num, height, width)

        Returns:
            A torch.Tensor with shape (batch_size, classes_num).
        """
        return self.layers(inputs)
