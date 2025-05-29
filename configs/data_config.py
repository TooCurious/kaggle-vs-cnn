import os

import torch
from easydict import EasyDict
from torchvision.transforms.v2 import Resize, Normalize, Compose, RandomHorizontalFlip, ColorJitter, \
    RandomResizedCrop, RandomRotation, CenterCrop, ToImage, ToDtype, RandAugment, RandomErasing, GaussianNoise

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_cfg = EasyDict()

data_cfg.emotion_detection = EasyDict()
data_cfg.emotion_detection.name = 'EmotionsDataset'  # Dataset class name

# Path to the directory with dataset files
data_cfg.emotion_detection.path_to_data = os.path.join(ROOT_DIR, 'data', 'emotion_detection')
data_cfg.emotion_detection.annot_filename = 'data_info.csv'

# Label mapping
data_cfg.emotion_detection.label_mapping = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}
data_cfg.emotion_detection.classes_num = 7
data_cfg.emotion_detection.channels_num = 1

# Training configuration
data_cfg.emotion_detection.train_transforms = Compose([
    ToImage(),
    ToDtype(torch.uint8, scale=True),
    # Resize((32, 32)),  # If RandomResizedCrop is not used
    RandomResizedCrop(size=48, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    RandomHorizontalFlip(),
    RandomRotation(degrees=10),
    ColorJitter(brightness=(0.5, 2), contrast=0.05),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=(0.5,), std=(0.5,))
])
data_cfg.emotion_detection.eval_transforms = Compose([
    ToImage(),
    # ToDtype(torch.uint8, scale=True),
    # Resize((32, 32)),  # If RandomResizedCrop is not used
    ToDtype(torch.float32, scale=True),
    Normalize(mean=(0.5,), std=(0.5,)),
])

data_cfg.resisc45 = EasyDict()
data_cfg.resisc45.name = 'Resisc45Dataset'  # Dataset class name

# Path to the directory with dataset files
data_cfg.resisc45.path_to_data = os.path.join(ROOT_DIR, 'data', 'RESISC45')
data_cfg.resisc45.annot_filename = 'data_info.csv'

# Label mapping
data_cfg.resisc45.label_mapping = {
    'baseball_diamond': 0, 'basketball_court': 1, 'beach': 2, 'circular_farmland': 3, 'forest': 4,
    'ground_track_field': 5, 'harbor': 6, 'industrial_area': 7, 'intersection': 8, 'meadow': 9, 'mobile_home_park': 10,
    'mountain': 11, 'overpass': 12, 'palace': 13, 'railway_station': 14, 'rectangular_farmland': 15, 'river': 16,
    'roundabout': 17, 'sea_ice': 18, 'snowberg': 19, 'sparse_residential': 20, 'storage_tank': 21, 'wetland': 22
}
data_cfg.resisc45.classes_num = 23
data_cfg.resisc45.channels_num = 3

# Training configuration
data_cfg.resisc45.normalization_params = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # ImageNet stats
data_cfg.resisc45.train_transforms = Compose([
    ToImage(),
    ToDtype(torch.uint8, scale=True),
    RandomResizedCrop(224),
    ToDtype(torch.float32, scale=True),
    Normalize(**data_cfg.resisc45.normalization_params),
])
data_cfg.resisc45.eval_transforms = Compose([
    ToImage(),
    ToDtype(torch.uint8, scale=True),
    Resize(256),
    CenterCrop(224),
    ToDtype(torch.float32, scale=True),
    Normalize(**data_cfg.resisc45.normalization_params),
])
