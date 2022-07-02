import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch

from src.models.clf.net import get_network
from src.conf.config import ClfModelConfig
from src.train_utils.data import get_loaders
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error


def get_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model_checkpoint_fp', type=pathlib.Path)
    argument_parser.add_argument('--split_dir', type=pathlib.Path)
    argument_parser.add_argument('--batch_size', type=int, default=28, required=False)
    argument_parser.add_argument('--num_workers', type=int, default=2, required=False)
    argument_parser.add_argument('--device', type=torch.device, default=torch.device('cuda'), required=False)

    return argument_parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    all_pred_ages = []
    all_pred_classes = []
    all_target_ages = []
    all_target_classes = []

    model = get_network(ClfModelConfig.n_classes, args.model_checkpoint_fp).to(args.device)
    model = model.eval()

    _, val_loader, test_loader = get_loaders(args.split_dir, args.batch_size, args.num_workers)
    for i, (images, target_classes, target_ages) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images = images.permute(0, 3, 1, 2).to(args.device).float()
        target_classes = target_classes.to(args.device).float()
        target_ages = target_ages.to(args.device).float()
        with torch.no_grad():
            features = model.get_features(images)
            pred_ages = model.get_age_prediction(features)
            pred_classes = model.get_clf_prediction(features)
        all_pred_ages.append(pred_ages)
        all_pred_classes.append(pred_classes)
        all_target_ages.append(target_ages)
        all_target_classes.append(target_classes)

    all_pred_classes = torch.concat(all_pred_classes).cpu().numpy().argmax(1)
    all_target_classes = torch.concat(all_target_classes).cpu().numpy().argmax(1)

    all_pred_ages = torch.concat(all_pred_ages).cpu().numpy()
    all_pred_ages = np.e ** (all_pred_ages[:, 0] * 4.45 - 0.01)
    all_target_ages = torch.concat(all_target_ages).cpu().numpy()
    all_target_ages = np.e ** (all_target_ages * 4.45 - 0.01)

    cm = confusion_matrix(all_target_classes, all_pred_classes)
    class_titles = val_loader.dataset.binarizer.classes_
    cm = pd.DataFrame(cm, columns=[class_titles], index=class_titles)
    age_mse = np.around(mean_squared_error(all_target_ages, all_pred_ages), 2)
    print(age_mse)

    # plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='.0f', ax=axs[0])
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=30, ha='right')
    axs[0].set_title('Матрица ошибок')

    axs[1].scatter(all_pred_ages, all_target_ages, label=f'mse: {str(age_mse)}')
    axs[1].set_title('Возраст')
    axs[1].set_xlabel('Реальный')
    axs[1].set_ylabel('Предсказанный')
    axs[1].legend()
    fig.tight_layout()

    print(classification_report(all_target_classes, all_pred_classes))
    plt.show()
