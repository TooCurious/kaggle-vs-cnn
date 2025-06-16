from easydict import EasyDict

from utils.enums import LayerType, WeightsInitType

model_cfg = EasyDict()
model_cfg.name = 'ResNetTricks'  # Define the model class (from ('CNN', 'ResNet', 'ResNetTricks'))

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

# ResNet
model_cfg.resnet = EasyDict()
model_cfg.resnet.depth = 18  # From (18, 34, 50, 101, 152)
model_cfg.resnet.stem = 64  # None if the stem module is skipped
model_cfg.resnet.expansion_factor = 1  # 1 for ResNet-18/34, 4 for ResNet-50/101/152
model_cfg.resnet.building_block = {
    18: LayerType.BaseBlock, 34: LayerType.BaseBlock, 50: LayerType.BottleneckBlock, 101: LayerType.BottleneckBlock,
    152: LayerType.BottleneckBlock
}[model_cfg.resnet.depth]
model_cfg.resnet.layers = {
    'stages': [
        {'type': model_cfg.resnet.building_block, 'specs': dict(block_out_channels=64, stride=1), 'block_repeats': 2},
        {'type': model_cfg.resnet.building_block, 'specs': dict(block_out_channels=128, stride=2), 'block_repeats': 2},
        {'type': model_cfg.resnet.building_block, 'specs': dict(block_out_channels=256, stride=2), 'block_repeats': 2},
        {'type': model_cfg.resnet.building_block, 'specs': dict(block_out_channels=512, stride=2), 'block_repeats': 2},
    ]
}

# ResNet with tricks
model_cfg.resnet_tricks = EasyDict()
model_cfg.resnet_tricks.depth = 18  # From (18, 34, 50, 101, 152)
model_cfg.resnet_tricks.pre_act = False  # If to use building blocks with full pre-activation
model_cfg.resnet_tricks.stem = 64  # None if the stem module is skipped
model_cfg.resnet_tricks.expansion_factor = 1  # 1 for ResNet-18/34, 4 for ResNet-50/101/152
model_cfg.resnet_tricks.building_block = {
    (18, False): LayerType.BaseBlock, (34, False): LayerType.BaseBlock, (50, False): LayerType.BottleneckBlock,
    (101, False): LayerType.BottleneckBlock, (152, False): LayerType.BottleneckBlock,
    (18, True): LayerType.BaseBlockFullPreAct, (34, True): LayerType.BaseBlockFullPreAct,
    (50, True): LayerType.BottleneckBlockFullPreAct, (101, True): LayerType.BottleneckBlockFullPreAct,
    (152, True): LayerType.BottleneckBlockFullPreAct
}[(model_cfg.resnet_tricks.depth, model_cfg.resnet_tricks.pre_act)]
model_cfg.resnet_tricks.layers = {
    'stages': [
        {
            'type': model_cfg.resnet_tricks.building_block,
            'specs': dict(block_out_channels=64, stride=1, trick_downsample=True, zero_gamma=True), 'block_repeats': 2
            # For full pre-activation blocks
            # 'specs': dict(block_out_channels=64, stride=1, after_stem=True, trick_downsample=True), 'block_repeats': 2
        },
        {
            'type': model_cfg.resnet_tricks.building_block,
            'specs': dict(block_out_channels=128, stride=2, trick_downsample=True, zero_gamma=True), 'block_repeats': 2
            # For full pre-activation blocks
            # 'specs': dict(block_out_channels=64, stride=1, after_stem=False, trick_downsample=True), 'block_repeats': 2
        },
        {
            'type': model_cfg.resnet_tricks.building_block,
            'specs': dict(block_out_channels=256, stride=2, trick_downsample=True, zero_gamma=True), 'block_repeats': 2
            # For full pre-activation blocks
            # 'specs': dict(block_out_channels=64, stride=1, after_stem=False, trick_downsample=True), 'block_repeats': 2
        },
        {
            'type': model_cfg.resnet_tricks.building_block,
            'specs': dict(block_out_channels=512, stride=2, trick_downsample=True, zero_gamma=True), 'block_repeats': 2
            # For full pre-activation blocks
            # 'specs': dict(block_out_channels=64, stride=1, after_stem=False, trick_downsample=True), 'block_repeats': 2
        },
    ]
}
model_cfg.resnet_tricks.trick_stem = True
model_cfg.resnet_tricks.scheduler = True
model_cfg.resnet_tricks.warmup = 0.1
model_cfg.resnet_tricks.no_bias_decay = True
model_cfg.resnet_tricks.mixup = True
