import torch
from src.trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    def update_metrics(self, preds, targets):
        preds = preds.detach().cpu()
        preds = torch.sigmoid(preds)

        targets = targets.detach().cpu().int()

        self.metrics['ClfAUROC'].update(preds, targets)
        self.metrics['ClfConfusionMatrix'].update(preds, targets)

    def log_metrics(self, series: str, iteration: int):
        self.logger.report_scalar('ClfAUROC', series, float(self.metrics['ClfAUROC'].compute()), iteration)
        self.logger.report_confusion_matrix('ClfConfusionMatrix', series,
                                            self.metrics['ClfConfusionMatrix'].compute(), iteration)

    def transform_batch_data(self, batch_data):
        images, _, _, is_norm = batch_data
        images = images.permute(0, 3, 1, 2).to(self.device).float()
        is_norm = torch.reshape(is_norm, (-1, 1)).to(self.device).float()
        return images, is_norm

    def get_predictions(self, inputs):
        features = self.model.get_features(inputs)
        clf_pred = self.model.get_clf_prediction(features)
        return clf_pred

    def compute_loss(self, preds, targets):
        return self.criterion(preds, targets)
