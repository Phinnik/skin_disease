import pathlib
from src.trainer.clf_trainer import Trainer
from src.models.clf.net import get_network
from src.train_utils.reproducibility import set_seed
from src.train_utils.data import get_loaders
from src.loss.wrapper import UncertaintyLossWrapper
from src.conf.config import ProjectPaths, ClfModelConfig
import torchmetrics
import clearml
import torch
import argparse


def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--seed_value', default=123123, required=False)
    argument_parser.add_argument('--n_classes', default=ClfModelConfig.n_classes, required=False)
    argument_parser.add_argument('--task_name', default='sample_task', required=False)
    argument_parser.add_argument('--n_epochs', default=20, required=False)
    argument_parser.add_argument('--device', default=torch.device('cuda'), type=torch.device, required=False)
    argument_parser.add_argument('--split_dir',
                                 default=ProjectPaths.data_dir.joinpath('processed/splits/skin_cancer_mnist_split'),
                                 type=pathlib.Path, required=False)
    argument_parser.add_argument('--batch_size', default=28, required=False)
    argument_parser.add_argument('--num_workers', default=3, required=False)
    argument_parser.add_argument('--lr', default=0.0001, required=False)
    argument_parser.add_argument('--one_cycle_max_lr', default=0.005, required=False)
    argument_parser.add_argument('--preprocessed_save_dir',
                                 default=ProjectPaths.data_dir.joinpath('external', 'preprocessed'),
                                 type=pathlib.Path, required=False)

    return argument_parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed_value)
    model = get_network(args.n_classes).to(args.device)
    train_loader, val_loader, test_loader = get_loaders(args.split_dir,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        preprocessed_save_dir=args.preprocessed_save_dir)
    criterion = UncertaintyLossWrapper([
        torch.nn.BCEWithLogitsLoss(), torch.nn.MSELoss()
    ])
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    metrics = torchmetrics.MetricCollection({
        'AgeMSE': torchmetrics.MeanSquaredError(),
        'ClfAUROC': torchmetrics.AUROC(num_classes=args.n_classes),
        'ClfConfusionMatrix': torchmetrics.ConfusionMatrix(args.n_classes)
    })
    task = clearml.Task.init('skin_disease', args.task_name)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.one_cycle_max_lr,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=args.n_epochs)

    trainer = Trainer(model, optimizer, criterion, metrics, train_loader,
                      val_loader, task, args.device, 20, args.n_epochs,
                      ProjectPaths.models_dir.joinpath(args.task_name), scheduler)
    trainer.run()
