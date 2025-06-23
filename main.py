import sys

import pandas as pd
from torch.utils.data import DataLoader

from configs.experiment_config import experiment_cfg
from dataset.emotions_dataset import EmotionsDataset
from dataset.resisc45 import Resisc45Dataset
from executors.trainer import Trainer
from utils.enums import SetType
from utils.visualization import show_batch


def train():
    trainer = Trainer(experiment_cfg)

    # One batch overfitting
    # trainer.batch_overfit()

    # Model training
    trainer.fit()


def predict():
    trainer = Trainer(experiment_cfg, init_logger=False)
    dataset = getattr(sys.modules[__name__], experiment_cfg.data.name)

    # Get data to make predictions on
    test_dataset = dataset(experiment_cfg.data, SetType.test, transforms=experiment_cfg.data.eval_transforms)
    test_dataloader = DataLoader(test_dataset, experiment_cfg.train.batch_size, shuffle=False)

    # Get predictions
    model_path = experiment_cfg.best_checkpoint_name
    predictions, image_paths = trainer.predict(model_path, test_dataloader)

    # Save results to submission file
    test_results_df = pd.DataFrame({'ID': image_paths, 'prediction': predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


def show_augmentations():
    # Don't forget to turn off normalization before visualizing!
    data_config = experiment_cfg.data
    dataset = getattr(sys.modules[__name__], data_config.name)
    train_dataset = dataset(data_config, SetType.train, transforms=data_config.eval_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    train_dataset_augmented = dataset(data_config, SetType.train, transforms=data_config.train_transforms)
    train_augmented_dataloader = DataLoader(train_dataset_augmented, batch_size=16, shuffle=True)

    random_batch = next(iter(train_dataloader))
    show_batch(random_batch)

    random_batch = next(iter(train_augmented_dataloader))
    show_batch(random_batch)


if __name__ == '__main__':
    train()
    # predict()
    # show_augmentations()
