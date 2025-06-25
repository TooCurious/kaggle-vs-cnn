import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union

import mlflow
import neptune
from dotenv import load_dotenv
from neptune.utils import stringify_unsupported


class BaseLogger(ABC):
    """A base experiment logger class."""

    @abstractmethod
    def __init__(self, config):
        """Logs git commit id, dvc hash, environment."""
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict):
        pass

    @abstractmethod
    def save_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self):
        pass


class NeptuneLogger(BaseLogger):
    """A neptune.ai experiment logger class."""

    def __init__(self, config):
        super().__init__(config)

        self.run = neptune.init_run(
            project=config.project,
            api_token=os.environ['NEPTUNE_API_TOKEN'],
            name=config.experiment_name,
            dependencies=config.dependencies_path,
            with_id=config.run_id
        )

    def log_hyperparameters(self, params: dict):
        """Model hyperparameters logging."""
        self.run['hyperparameters'] = stringify_unsupported(params)

    def save_metrics(self, type_set, metric_name: Union[list[str], str], metric_value: Union[list[float], float],
                     step=None):
        if isinstance(metric_name, list):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].log(p_v, step=step)
        else:
            self.run[f"{type_set}/{metric_name}"].log(metric_value, step=step)

    def save_plot(self, type_set, plot_name, plt_fig):
        self.run[f"{type_set}/{plot_name}"].append(plt_fig)

    def stop(self):
        self.run.stop()


class MLFlowLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

        self._init_experiment(config)

    def _init_experiment(self, config):
        """Sets up experiment configurations to log."""
        mlflow.set_experiment(config.experiment_name or "default_experiment")

        if config.run_id:
            mlflow.start_run(run_id=config.run_id)
            metrics_dir = f"./mlruns/{mlflow.get_run(config.run_id).info.experiment_id}/{config.run_id}/metrics"
            self.steps = {
                metric_name: len(open(os.path.join(metrics_dir, metric_name)).readlines())
                for metric_name in os.listdir(metrics_dir)
            }
        else:
            mlflow.start_run()
            self.steps = defaultdict(lambda: 0)

        mlflow.log_artifact(config.dependencies_path)

    def log_hyperparameters(self, params: dict):
        mlflow.log_params(params)

    def save_metrics(self, type_set, metric_name: Union[list[str], str], metric_value: Union[list[float], float],
                     step=None):
        if isinstance(metric_name, list):
            for p_n, p_v in zip(metric_name, metric_value):
                key = f"{type_set}_{p_n}"
                mlflow.log_metric(key, p_v, step if step else self.steps[key])
                self.steps[key] += 1
        else:
            key = f"{type_set}_{metric_name}"
            mlflow.log_metric(key, metric_value, step if step else self.steps[key])
            self.steps[key] += 1

    def save_plot(self, type_set, plot_name, plt_fig):
        mlflow.log_figure(plt_fig, f"{type_set}_{plot_name}.png")

    def stop(self):
        mlflow.end_run()
