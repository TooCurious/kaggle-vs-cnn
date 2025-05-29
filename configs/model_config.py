from easydict import EasyDict

from utils.enums import LayerType, WeightsInitType

model_cfg = EasyDict()
model_cfg.name = 'CNN'  # Define the model class

# Weights and bias initialization
model_cfg.params = EasyDict()

model_cfg.params.convolutional = EasyDict()
model_cfg.params.convolutional.init_type = WeightsInitType.kaiming_normal
model_cfg.params.convolutional.init_kwargs = {'mode': 'fan_out', 'nonlinearity': 'relu'}
model_cfg.params.convolutional.zero_bias = True

model_cfg.params.linear = EasyDict()
model_cfg.params.linear.init_type = WeightsInitType.kaiming_normal
model_cfg.params.linear.init_kwargs = {'mode': 'fan_in', 'nonlinearity': 'relu'}
model_cfg.params.linear.zero_bias = True

# Base model
model_cfg.cnn = EasyDict()
model_cfg.cnn.block_params = dict(conv_kernel=3, conv_padding=1, conv_stride=1, max_pool_kernel=2, max_pool_stride=2)
model_cfg.cnn.layers = {
    'features': [
        {
            'type': LayerType.BaseBlock,
            'specs': dict(in_channels=..., out_channels=64, **model_cfg.cnn.block_params, conv_num=2)
        },
        {
            'type': LayerType.BaseBlock,
            'specs': dict(in_channels=64, out_channels=128, **model_cfg.cnn.block_params, conv_num=2)
        },
        {
            'type': LayerType.BaseBlock,
            'specs': dict(in_channels=128, out_channels=256, **model_cfg.cnn.block_params, conv_num=3)
        },
        {
            'type': LayerType.BaseBlock,
            'specs': dict(in_channels=256, out_channels=512, **model_cfg.cnn.block_params, conv_num=3)
        },
        {
            'type': LayerType.BaseBlock,
            'specs': dict(in_channels=512, out_channels=512, **model_cfg.cnn.block_params, conv_num=3)
        },
    ],
    'classifier': [
        {'type': LayerType.Flatten, 'specs': dict()},
        # To set in_features parameter, you need to calculate image size transformation during model 'features' part
        {'type': LayerType.Linear, 'specs': dict(in_features=512 * 7 * 7, out_features=4096)},
        {'type': LayerType.ReLU, 'specs': dict()},
        {'type': LayerType.Linear, 'specs': dict(in_features=4096, out_features=4096)},
        {'type': LayerType.ReLU, 'specs': dict()},
        {'type': LayerType.Linear, 'specs': dict(in_features=4096, out_features=...)},
    ]
}
