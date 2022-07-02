import pathlib
import clearml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchmetrics
import abc


class BaseTrainer(abc.ABC):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 metrics: torchmetrics.MetricCollection,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 clearml_task: clearml.Task,
                 device: torch.device,
                 log_frequency: int,
                 num_epochs: int,
                 models_save_dir: pathlib.Path = None,
                 lr_scheduler=None):
        """
        Class for models training

        :param model: pytorch models
        :param optimizer: pytorch optimizer
        :param criterion: loss criterion
        :param metrics: set of metrics wrapped in torchmetrics.MetricCollection
        :param train_loader: DataLoader for training stage
        :param val_loader: DataLoader for validation stage
        :param clearml_task: clearml.Task class of experiment
        :param device: torch.device to use
        :param log_frequency: iterations frequency to log metrics, losses, learning rate etc.
        :param num_epochs: number of epochs to train
        :param models_save_dir: directory to save best and last models
        :param lr_scheduler: learning rate scheduler
        """

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.metrics = metrics
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clearml_task = clearml_task
        self.device = device
        self.log_frequency = log_frequency
        self.num_epochs = num_epochs
        self.clearml_task = clearml_task
        self.logger = clearml_task.get_logger()
        self.models_save_dir = models_save_dir
        self.min_loss = float('inf')

    def log_metrics(self, series: str, iteration: int):
        """
        Is called every self.log_frequency and logs self.metrics to ClearMl

        :param series: e.g. {'train', 'validation'}
        :param iteration: batch number of report
        """
        for metric_name, value in self.metrics.compute().items():
            if len(value.shape) == 0:
                self.logger.report_scalar(metric_name, series, value, iteration)
            elif len(value.shape) > 1:
                self.logger.report_confusion_matrix(metric_name, series, value.numpy(), iteration)

    @abc.abstractmethod
    def update_metrics(self, preds, targets):
        """
        Is called every batch to update running metrics

        :param preds: models predictions
        :param targets: batch_targets
        """
        pass

    @abc.abstractmethod
    def transform_batch_data(self, batch_data) -> tuple:
        """
        Is called to transform batch data after loading it from dataloader.
        Should return suitable data for training.
        """
        pass

    @abc.abstractmethod
    def get_predictions(self, inputs):
        """
        Is called when models prediction is needed.

        :param inputs: models inputs
        """
        pass

    @abc.abstractmethod
    def compute_loss(self, preds, targets):
        """
        Is called to compute loss from models prediction and batch targets

        :param preds: models prediction
        :param targets: batch targets
        """
        pass

    def train_one_epoch(self, epoch_idx: int):
        """
        Runs training for one epoch

        :param epoch_idx: epoch index
        """
        self.metrics.reset()
        self.model.train()
        running_loss = 0
        titer = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='training')
        for batch_idx, batch_data in titer:
            inputs, targets = self.transform_batch_data(batch_data)

            self.optimizer.zero_grad()
            preds = self.get_predictions(inputs)
            loss = self.compute_loss(preds, targets)
            loss.backward()
            self.optimizer.step()
            self.update_metrics(preds, targets)

            running_loss += loss.item()
            iteration = epoch_idx * len(self.train_loader) + batch_idx

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if (iteration % self.log_frequency == 0) & (batch_idx != 0) | (batch_idx == (len(self.train_loader) - 1)):
                self.log_metrics('train', iteration)
                self.logger.report_scalar('loss', 'train', running_loss / (batch_idx + 1), iteration)

        self.logger.report_scalar('Learning rate',
                                  'train',
                                  self.optimizer.param_groups[0]['lr'],
                                  (epoch_idx + 1) * len(self.train_loader))

    def validate_one_epoch(self, epoch_idx: int):
        """
        Validates models for one epoch

        :param epoch_idx: epoch index
        """
        self.metrics.reset()
        self.model.eval()
        running_loss = 0
        titer = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='validating')
        for batch_idx, batch_data in titer:
            inputs, targets = self.transform_batch_data(batch_data)
            with torch.no_grad():
                preds = self.get_predictions(inputs)
            loss = self.compute_loss(preds, targets)

            running_loss += loss.item()
            self.update_metrics(preds, targets)

        iteration = (epoch_idx + 1) * len(self.train_loader)
        self.log_metrics('validation', iteration)
        self.logger.report_scalar('loss', 'validation', running_loss / len(self.val_loader), iteration)

        if self.models_save_dir is not None:
            self.models_save_dir.mkdir(exist_ok=True, parents=True)
            torch.save(self.model, self.models_save_dir.joinpath('model_last.pkl'))
            if running_loss / len(self.val_loader) < self.min_loss:
                torch.save(self.model, self.models_save_dir.joinpath('model_best.pkl'))

    def run(self):
        """
        Runs training
        """
        for epoch_idx in tqdm(range(self.num_epochs)):
            self.train_one_epoch(epoch_idx)
            self.validate_one_epoch(epoch_idx)
