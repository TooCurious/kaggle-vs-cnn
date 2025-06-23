import os

from easydict import EasyDict

from configs.data_config import data_cfg
from configs.model_config import model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_cfg = EasyDict()
experiment_cfg.seed = 0
experiment_cfg.epochs_num = 200

# Train parameters
experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 64
experiment_cfg.train.learning_rate = 0.001
experiment_cfg.train.continue_train = False
experiment_cfg.train.checkpoint_from_epoch = None
experiment_cfg.train.inference_frequency = 2
experiment_cfg.train.label_smoothing = 0
experiment_cfg.train.weight_decay = 0

# Overfit parameters
experiment_cfg.overfit = EasyDict()
experiment_cfg.overfit.iterations_num = 500

# Neptune parameters
experiment_cfg.neptune = EasyDict()
experiment_cfg.neptune.env_path = os.path.join(ROOT_DIR, '.env')
experiment_cfg.neptune.project = 'CNN-research'
experiment_cfg.neptune.experiment_name = 'test_4'
experiment_cfg.neptune.run_id = None
experiment_cfg.neptune.dependencies_path = os.path.join(ROOT_DIR, 'requirements.txt')

# Checkpoints parameters
experiment_cfg.checkpoints_dir = os.path.join(
    ROOT_DIR, 'experiments', experiment_cfg.neptune.experiment_name, 'checkpoints'
)
experiment_cfg.checkpoint_save_frequency = 1
experiment_cfg.checkpoint_name = 'checkpoint_%s'
experiment_cfg.best_checkpoint_name = 'best_checkpoint'

# Data parameters (choose needed dataset)
experiment_cfg.data = data_cfg.resisc45

# Model parameters
experiment_cfg.model = model_cfg
