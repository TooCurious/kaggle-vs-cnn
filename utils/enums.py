from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'validation', 'test'))
LayerType = IntEnum(
    'LayerType', (
        'Linear', 'ReLU', 'Dropout', 'Flatten', 'BaseBlock', 'BottleneckBlock', 'BaseBlockFullPreAct',
        'BottleneckBlockFullPreAct'
    )
)
WeightsInitType = IntEnum(
    'WeightsInitType', ('normal', 'uniform', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
)
