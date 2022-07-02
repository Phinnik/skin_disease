import torch
from src.trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    def update_metrics(self, preds, targets):
        pred_class, pred_age = preds
        target_class, target_age = targets

        pred_class = pred_class.detach().cpu()
        pred_class = torch.nn.functional.softmax(pred_class, dim=1)
        pred_age = pred_age.detach().cpu()

        target_class = target_class.detach().cpu().int()
        target_class = torch.where(target_class)[1]
        target_age = target_age.detach().cpu()

        self.metrics['AgeMSE'].update(pred_age, target_age)
        self.metrics['ClfAUROC'].update(pred_class, target_class)
        self.metrics['ClfConfusionMatrix'].update(pred_class, target_class)

    def log_metrics(self, series: str, iteration: int):
        self.logger.report_scalar('AgeMSE', series, float(self.metrics['AgeMSE'].compute()), iteration)
        self.logger.report_scalar('ClfAUROC', series, float(self.metrics['ClfAUROC'].compute()), iteration)
        self.logger.report_confusion_matrix('ClfConfusionMatrix', series,
                                            self.metrics['ClfConfusionMatrix'].compute(), iteration)

    def transform_batch_data(self, batch_data):
        images, class_ecnoding, age = batch_data
        images = images.permute(0, 3, 1, 2).to(self.device).float()
        class_ecnoding = class_ecnoding.squeeze().to(self.device).float()
        age = age[:, None].to(self.device).float()
        return images, (class_ecnoding, age)

    def get_predictions(self, inputs):
        features = self.model.get_features(inputs)
        age = self.model.get_age_prediction(features)
        clf_pred = self.model.get_clf_prediction(features)
        return clf_pred, age

    def compute_loss(self, preds, targets):
        return self.criterion(preds, targets)
