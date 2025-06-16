import os
import sys

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import MixUp
from tqdm import tqdm

from dataloaders.batch_samplers import Upsampling
from dataset.emotions_dataset import EmotionsDataset
from dataset.resisc45 import Resisc45Dataset
from models.cnn import CNN
from models.resnet import ResNet
from models.resnet_tricks import ResNetTricks
from utils.common_functions import set_seed
from utils.enums import SetType
from utils.logger import MLFlowLogger, NeptuneLogger
from utils.metrics import accuracy_score, confusion_matrix
from utils.visualization import plot_confusion_matrix


class Trainer:
    """A class for model training."""

    def __init__(self, config, init_logger: bool = True):
        self.config = config
        set_seed(self.config.seed)

        self._prepare_data()
        self._prepare_model()
        self._add_tricks()

        self._init_logger(init_logger)

    def _init_logger(self, init_logger: bool):
        if init_logger:
            self.logger = NeptuneLogger(self.config.neptune)
            if not self.config.train.continue_train:
                self.logger.log_hyperparameters(self.config)

    def _prepare_data(self):
        """Prepares training and validation data."""
        data_cfg = self.config.data
        dataset = getattr(sys.modules[__name__], data_cfg.name)
        batch_size = self.config.train.batch_size

        train_transforms = data_cfg.train_transforms
        validation_transforms = data_cfg.eval_transforms

        self.train_dataset = dataset(data_cfg, SetType.train, transforms=train_transforms)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size, drop_last=True, sampler=Upsampling(self.train_dataset),
            num_workers=os.cpu_count()
        )

        eval_train_dataset = dataset(data_cfg, SetType.train, transforms=validation_transforms)
        self.eval_train_dataloader = DataLoader(
            eval_train_dataset, batch_size, shuffle=False, num_workers=os.cpu_count()
        )

        self.validation_dataset = dataset(data_cfg, SetType.validation, transforms=validation_transforms)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

    def _add_tricks(self):
        if not self.config.model.name == 'ResNetTricks':
            return

        if self.config.model.resnet_tricks.no_bias_decay:
            no_decay, with_decay = [], []

            for m in self.model.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    with_decay.append(m.weight)
                    if m.bias is not None:
                        no_decay.append(m.bias)
                else:
                    if hasattr(m, 'weight'):
                        no_decay.append(m.weight)
                    if hasattr(m, 'bias'):
                        no_decay.append(m.bias)

            params_groups = [
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': with_decay, 'weight_decay': self.config.train.weight_decay}
            ]
            self.optimizer = torch.optim.SGD(params_groups, lr=self.config.train.learning_rate)

        if self.config.model.resnet_tricks.scheduler:
            training_steps = self.config.epochs_num * len(self.train_dataloader)
            self.warmup_steps = self.config.model.resnet_tricks.warmup * training_steps
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: step / self.warmup_steps
            )
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=training_steps - self.warmup_steps
            )

        if self.config.model.resnet_tricks.mixup:
            self.mixup = MixUp(num_classes=self.config.data.classes_num)

    def _prepare_model(self):
        """Prepares model, optimizer and loss function."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_class = getattr(sys.modules[__name__], self.config.model.name)
        data_specs = dict(in_channels=self.config.data.channels_num, classes_num=self.config.data.classes_num)
        self.model = model_class(self.config.model, **data_specs).to(self.device)
        self.config.model.scheme = str(self.model.layers)

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.config.train.learning_rate, weight_decay=self.config.train.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.train.label_smoothing)

    def save(self, filepath: str):
        """Saves trained model."""
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        state_dicts = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        if hasattr(self, 'warmup_scheduler') and hasattr(self, 'cosine_scheduler'):
            state_dicts['warmup_scheduler_state_dict'] = self.warmup_scheduler.state_dict()
            state_dicts['cosine_scheduler_state_dict'] = self.cosine_scheduler.state_dict()

        torch.save(state_dicts, os.path.join(self.config.checkpoints_dir, filepath))

    def load(self, filepath: str):
        """Loads trained model."""
        checkpoint = torch.load(os.path.join(self.config.checkpoints_dir, filepath), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'warmup_scheduler') and hasattr(self, 'cosine_scheduler'):
            self.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
            self.cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])

    def update_best_params(self, valid_metric: float, best_metric: float) -> float:
        """Updates best parameters: saves model if metrics exceeds the best values achieved."""
        if best_metric < valid_metric:
            self.save(self.config.best_checkpoint_name)
            best_metric = valid_metric
        return best_metric

    def make_step(self, batch: dict, epoch: int = None, step: int = None, update_model: bool = False) \
            -> (float, np.ndarray):
        """This method performs one step, including forward pass, calculation of the target function, backward
        pass and updating the model weights (if update_model is True).

        Args:
            batch: A batch data dictionary.
            epoch: The current epoch (if training).
            step: The current epoch's step (if training).
            update_model (bool): If True it is necessary to perform a backward pass and update the model weights.

        Returns:
            loss: The loss function value.
            output: The model output (batch_size x classes_num).
        """
        images = batch['image'].to(self.device)
        targets = batch['target'].to(self.device)

        if update_model and getattr(self, 'mixup', None):
            images, targets = self.mixup(images, targets)

        output = self.model(images)
        loss = self.criterion(output, targets)

        if update_model:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.config.model.name == 'ResNetTricks' and self.config.model.resnet_tricks.scheduler:
                if epoch * len(self.train_dataloader) + step <= self.warmup_steps:
                    self.warmup_scheduler.step()
                else:
                    self.cosine_scheduler.step()
                self.logger.save_metrics(SetType.train.name, 'lr', self.optimizer.param_groups[0]['lr'])

        return loss.item(), output.detach().cpu().numpy()

    def train_epoch(self, epoch: int):
        """Trains the model on training data for one epoch.

        The method goes through all train_dataloader batches and calls the self.make_step() method at each step.
        """
        self.model.train()
        pbar = tqdm(self.train_dataloader)

        for step, batch in enumerate(pbar):
            loss, output = self.make_step(batch, epoch=epoch, step=step, update_model=True)

            predictions = output.argmax(axis=-1)
            accuracy = accuracy_score(batch['target'].numpy(), predictions)

            self.logger.save_metrics(SetType.train.name, 'loss', loss)
            self.logger.save_metrics(SetType.train.name, 'accuracy', accuracy)
            pbar.set_description(f'Loss: {loss:.4f}, Train accuracy: {accuracy:.4f}')

    def fit(self):
        """The main model training loop."""
        best_metric = 0
        start_epoch = 0

        if self.config.train.continue_train:
            epoch = self.config.train.checkpoint_from_epoch
            self.load(self.config.checkpoint_name % epoch)
            start_epoch = epoch + 1

        for epoch in range(start_epoch, self.config.epochs_num):
            self.train_epoch(epoch)

            if epoch % self.config.train.inference_frequency == 0:
                self.evaluate(epoch, self.eval_train_dataloader, SetType.train)
                valid_metric = self.evaluate(epoch, self.validation_dataloader, SetType.validation)
                best_metric = self.update_best_params(valid_metric, best_metric)

            if epoch % self.config.checkpoint_save_frequency == 0:
                self.save(self.config.checkpoint_name % epoch)

    @torch.no_grad()
    def evaluate(self, epoch: int, dataloader: DataLoader, set_type: SetType) -> float:
        """Evaluation.

        The method is used to make the model performance evaluation on training/validation/test data.

        Args:
            epoch: A current training epoch.
            dataloader: The dataloader for the chosen set type.
            set_type: The set type chosen to evaluate.
        """
        self.model.eval()

        total_loss = []
        all_outputs, all_targets = [], []

        for batch in dataloader:
            loss, output = self.make_step(batch, update_model=False)

            total_loss.append(loss)
            all_outputs.append(output)
            all_targets.append(batch['target'].numpy())

        total_loss = np.mean(total_loss)
        all_predictions = np.concatenate(all_outputs).argmax(axis=-1)
        all_targets = np.concatenate(all_targets)

        accuracy = accuracy_score(all_targets, all_predictions)

        self.logger.save_metrics('eval_' + set_type.name, 'loss', total_loss, step=epoch)
        self.logger.save_metrics('eval_' + set_type.name, 'accuracy', accuracy, step=epoch)

        if self.config.data.name == 'EmotionsDataset':
            cm_plot = plot_confusion_matrix(
                confusion_matrix(all_targets, all_predictions, self.config.data.classes_num),
                title=f'Confusion matrix (epoch {epoch})',
                class_labels=list(self.config.data.label_mapping.keys())
            )
            self.logger.save_plot('eval_' + set_type.name, 'confusion_matrix', cm_plot)

        return accuracy

    @torch.no_grad()
    def predict(self, model_path: str, dataloader: DataLoader) -> (list, list):
        """Gets model predictions for a given dataloader."""
        self.load(model_path)
        self.model.eval()
        all_outputs, all_image_paths = [], []

        for batch in dataloader:
            all_image_paths.extend(batch['path'])

            output = self.model(batch['image'].to(self.device))
            all_outputs.append(output)

        all_predictions = torch.cat(all_outputs).argmax(-1)

        return all_predictions.tolist(), all_image_paths

    def batch_overfit(self):
        """One batch overfitting.

        This feature can be useful for debugging and evaluating your model's ability to learn and update its weights.
        """
        self.model.train()
        batch = next(iter(self.train_dataloader))

        for _ in range(self.config.overfit.iterations_num):
            loss_value, output = self.make_step(batch, update_model=True)
            accuracy = accuracy_score(batch['target'].numpy(), output.argmax(-1))

            self.logger.save_metrics(SetType.train.name, 'loss', loss_value)
            self.logger.save_metrics(SetType.train.name, 'accuracy', accuracy)
